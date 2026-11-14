"""_ChunkCacheBase, the chunk store every terrain style's cache builds on."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import numpy as np

from descape import (
    composite_backend,
    debug_log,
    iso_geometry,
    native_composite,
    perf_trace,
    render_paint,
    terrain_blend,
)
from descape.grid_overlay import DEFAULT_GRID, GridBake
from descape.render_cache_common import (
    _BUILD_TIME_LAYER_FIELDS,
    _DRAW_ONLY_LAYER_FIELDS,
    STAMP_PRUNE_EVERY,
    ChunkJob,
    RectJob,
    _Reduced,
    join_chunk_jobs,
)
from descape.render_cache_mip import _ChunkCacheMipMixin
from descape.render_cache_patch import _ChunkCachePatchMixin
from descape.render_cache_units import _ChunkCacheUnitsMixin
from descape.unit_filter import UnitFilter
from descape.view_layers import LayerState, render_key


class _ChunkCacheBase(_ChunkCacheMipMixin, _ChunkCacheUnitsMixin, _ChunkCachePatchMixin):
    """Grid/LRU bookkeeping shared by IsoChunkCache (Stepped, Phase B-B) and
    FlatChunkCache (Flat, Phase B-E) -- extracted because get_chunk/
    render_rect/patch/invalidate_region are pure chunk-grid arithmetic with
    zero mode-specific content, proven correct by tools/verify_iso_chunks.py's
    own byte-identity checks well before this split existed. This is NOT a
    weakening of composite_rect_iso()/composite_rect_flat()'s own "stay an
    independent implementation, never re-expressed in terms of the chunk
    path" rule -- that rule is about the COMPOSITOR (only ever reached here
    through the subclass's _composite_rect() hook, still two genuinely
    separate functions); this class owns LRU/grid bookkeeping only, the same
    thing any other chunk cache would.

    A subclass must, in its own __init__ (kept fully subclass-owned, not
    called from here, so each cache's own constructor signature/docstring
    stays exactly as-is): set self.chunk_px, call self._init_mip_levels(...)
    (Phase B-D-a; the level table must exist before _init_max_chunks() can
    read canvas_dims()), call self._refresh_source_caches() once, then call
    self._init_max_chunks(chunk_px, max_chunks). It must also implement:
      - canvas_dims(mip=0) -> (width, height) in LEVEL canvas pixels
      - _composite_rect(mip, x0, y0, x1, y1) -> (h, w, 3) uint8 array
      - _refresh_source_caches() -> None
      - a `style` class attribute (one of terrain_style.TERRAIN_STYLES:
        "flat" / "stepped" / "sloped") -- checked at the
        MapView.set_source() boundary (Phase B-E) to catch a cache wired to
        the wrong terrain style at construction time, rather than only once
        an edit exposes the mismatch later.

    Coordinate-space convention (Phase B-D-a, not stated explicitly
    elsewhere): render_rect()/get_chunk()/canvas_dims() all take
    LEVEL pixels/indices -- a caller past this class (Phase B-D-c's paint())
    is expected to already know which level it's asking for. patch()/
    invalidate_region() instead take REFERENCE canvas pixels, because their
    only real callers (ViewerWindow._apply_dirty, via dirty_screen_bbox_iso
    and a locally-recomputed reference tile_px) have no notion of levels at
    all -- Track B-D-a/b deliberately never touch viewer.py. _bbox_to_level()
    is the one conversion point between the two spaces.

    mip is part of every cache key (Phase B-B); Phase B-D-a makes get_chunk()/
    render_rect() actually use it for the pixel math (previously always 0),
    and patch()/invalidate_region() fan out across every RESIDENT level
    (not every enumerated one -- see _init_mip_levels' docstring) instead of
    hardcoding chunk 0."""

    style: str = ""
    # P3-g. Per-INSTANCE state with a class-level default, so every subclass
    # has the attribute without each __init__ having to set it. A class default
    # is safe here only because it is an immutable bool that set_sprites_enabled
    # rebinds on the instance; never give a mutable one this treatment.
    sprites_enabled: bool = False
    # View > Grid's baked spec, read by every _composite_rect beside
    # self.layers. Same class-default reasoning: frozen, rebound per instance.
    grid: GridBake = DEFAULT_GRID
    # Bumped by every mutating entry point, so a ChunkJob composited before
    # one is never installed after it (install_chunk). Same immutable-default
    # reasoning as above.
    _mutation_epoch: int = 0
    # Whether _unit_pack_of() exists: Stepped and Sloped keep a native unit pack.
    _has_unit_pack: bool = False
    # Flat int64 terrain id and layer per tile (y * w + x), for the native
    # composite and the blend keys; every cache builds both in __init__.
    _terrain_ids: np.ndarray | None = None
    _layer_ids: np.ndarray | None = None
    # The blend key of every tile, memoized from those two mirrors (_refresh_terrain_ids()).
    _blend_key_map: terrain_blend.KeyMap | None = None
    # The pool render_rect() fans missing chunks out to (set_chunk_executor()); None runs serial.
    _chunk_executor = None
    # Unread today: kept for the plan's `workers >= 2` gate, which its 2-core bench did not need.
    _chunk_workers: int = 0
    # Fewest uncached chunks a render_rect() pools, or slices a patch pools; and keys per prepare/join/install batch.
    POOL_MIN_MISSING = 2
    POOL_BATCH = 64
    # Fewest summed slice tiles (level px / tile_px**2) a patch pools; TASK-031.67 step 0's 280,000 px at tile_px 16.
    POOL_MIN_PATCH_TILE_AREA = 1100
    # The garrison hosts this cache's unit sources reflect (_record_flag_hosts(), GH #203).
    _flag_hosts: frozenset[int] = frozenset()
    # The wall radian verdict this cache's unit sources reflect (_record_wall_verdict(), TASK-583).
    _wall_radian: bool | None = None

    def _init_max_chunks(self, chunk_px: int, max_chunks: int | None) -> None:
        """Whole-canvas-at-chunk_px default, not some smaller fixed
        constant: the viewer always fitInView()s the full map on open, so
        the default/steady-state working set IS every chunk. A smaller cap
        would silently thrash (evict chunks the very next full repaint
        needs again) instead of ever reaching a warm, blit-only steady
        state -- the exact failure mode a chunk cache exists to avoid. See
        each subclass's own docstring for the measured memory cost of this
        default.

        Phase B-D-a: max_chunks explicitly passed keeps capping by COUNT
        with no byte bound at all (preserves every existing caller that
        passes one, e.g. tools/verify_iso_chunks.py's max_chunks=2/100000
        eviction tests, with no test edits); max_chunks=None (the real
        app's default) switches to a BYTE budget sized at exactly today's
        admitted set (canvas_dims() area * 3), tracked incrementally in
        self._cache_bytes rather than recomputed per eviction check. A
        byte budget is not itself the mechanism that stops N resident mip
        levels multiplying memory -- with chunk_px constant in LEVEL
        pixels, a byte budget is approximately a count budget too; what
        actually bounds total memory is the single shared global LRU
        below, unchanged, evicting across every level's chunks under one
        shared bound. The byte form matters for exactness on the ragged
        last chunk at each level, and because "N levels redistribute
        memory, they don't multiply it" is a claim about bytes."""
        self.chunk_px = chunk_px
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        # Mips any chunk was ever inserted at; never cleared (level_warmed(), TASK-031.82 D3).
        self._composited_mips: set[int] = set()
        # The reduced store (render_reduced_rect()): per-instance, never class defaults.
        self._chunk_stamp: dict[tuple[int, int, int], int] = {}
        self._stamp_counter = 0
        self._reduced: OrderedDict[tuple[int, int, int, int], _Reduced] = OrderedDict()
        self._reduced_bytes = 0
        if max_chunks is None:
            canvas_w, canvas_h = self.canvas_dims()
            self.max_chunks: int | None = None
            self.max_bytes: int | None = canvas_w * canvas_h * 3
        else:
            self.max_chunks = max_chunks
            self.max_bytes = None

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        raise NotImplementedError

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        raise NotImplementedError

    def _refresh_source_caches(self, elevation_changed: set | None = None, splice_levels=None) -> None:
        """splice_levels: patch()'s rebuild_levels, the levels allowed to do
        per-level source work now. Only _LeveledChunkCache's elevation splice reads it."""
        raise NotImplementedError

    def _defers_patch(self, mip: int) -> bool:
        """Whether a resident level must evict patch()'s chunks rather than
        composite them: its unit layer has deferred work (_LeveledChunkCache)."""
        return False

    def chunk_index_range(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        """(cx0, cy0, cx1, cy1) -- the inclusive chunk-grid index range that
        LEVEL `mip` pixel rect [x0, x1) x [y0, y1) covers, at this cache's
        chunk_px. `mip` is accepted (unused here) purely so a call site that
        already has a level's mip alongside its rect can pass both without a
        second lookup -- chunk_px is the same at every level, so mip plays
        no part in the arithmetic.

        Written three times already, inline, as this same `x0 // chunk_px` /
        `(x1 - 1) // chunk_px` pair: render_rect(), patch(), and
        invalidate_region(). New code (2026-09-07 plan's A2/A3) calls this
        instead of a fourth copy. Those three existing sites are
        deliberately NOT rewired to it: render_rect() is the hottest
        byte-identity-guarded path in this cache, and patch()/
        invalidate_region() interleave this arithmetic with a
        reference-to-level conversion this function doesn't do -- rewiring
        either buys nothing here at a real risk of regressing code this
        module's own byte-identity tests already cover. The duplication is
        noted, not chased."""
        cx0, cy0 = x0 // self.chunk_px, y0 // self.chunk_px
        cx1, cy1 = (x1 - 1) // self.chunk_px, (y1 - 1) // self.chunk_px
        return cx0, cy0, cx1, cy1

    def _evict(self) -> None:
        """Evicts least-recently-used chunks while EITHER configured bound
        (see _init_max_chunks) is exceeded. A chunk is at most chunk_px**2
        * 3 bytes and is clipped to canvas bounds at the high edge, so a
        just-inserted chunk can never itself exceed max_bytes on a
        default-constructed cache -- this can't evict down to empty."""
        while self._cache and (
            (self.max_chunks is not None and len(self._cache) > self.max_chunks)
            or (self.max_bytes is not None and self._cache_bytes > self.max_bytes)
        ):
            _key, victim = self._cache.popitem(last=False)
            self._cache_bytes -= victim.nbytes

    def get_chunk(self, mip: int, cx: int, cy: int) -> np.ndarray:
        """Returns chunk (mip, cx, cy)'s composited pixels, from cache if
        present (moved to most-recently-used), else composited fresh via
        self._composite_rect() and inserted, evicting past either bound
        (see _evict). Clipped to canvas bounds at the high edge -- a chunk
        straddling the canvas edge is smaller than chunk_px x chunk_px,
        same "ragged last chunk" shape any tile-based grid has. cx/cy are
        LEVEL chunk-grid indices, sized against that level's own
        canvas_dims(mip)."""
        key = (mip, cx, cy)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        chunk = self._composite_rect(mip, *self._chunk_rect(mip, cx, cy))
        self._insert(key, chunk)
        return chunk

    def _chunk_rect(self, mip: int, cx: int, cy: int) -> tuple[int, int, int, int]:
        canvas_w, canvas_h = self.canvas_dims(mip)
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        return x0, y0, min(x0 + self.chunk_px, canvas_w), min(y0 + self.chunk_px, canvas_h)

    def _insert(self, key: tuple[int, int, int], chunk: np.ndarray) -> None:
        self._cache[key] = chunk
        self._composited_mips.add(key[0])
        self._cache_bytes += chunk.nbytes
        self._cache.move_to_end(key)
        self._stamp(key)
        self._evict()

    def _stamp(self, key: tuple[int, int, int]) -> None:
        """Marks chunk `key`'s pixels as newly set. _insert() and _write_slice()
        are the only writers of a cached chunk's pixels, and both call this, so
        a reduced entry built over the old pixels fails _reduced_valid()."""
        self._stamp_counter += 1
        self._chunk_stamp[key] = self._stamp_counter
        if self._stamp_counter % STAMP_PRUNE_EVERY == 0:
            cache = self._cache
            self._chunk_stamp = {k: v for k, v in self._chunk_stamp.items() if k in cache}

    def _prepare_rect(
        self, mip: int, x0: int, y0: int, x1: int, y1: int, heights: np.ndarray | None = None
    ) -> tuple | None:
        """render_bbox.prepare_rect_native() for this cache's sources, or None when
        the rect has no worker-safe path. Base (Flat): none. heights: a
        _heights_snapshot() the job reads as is, None for its own copy."""
        return None

    def prepare_chunk_job(self, mip: int, cx: int, cy: int, heights: np.ndarray | None = None) -> ChunkJob | None:
        """get_chunk() split for margin_warm's worker threads and a pooled
        render_rect(): everything but the kernel runs here, on the caller's
        (GUI) thread. None means call get_chunk() instead: the chunk is
        cached, or has no worker-safe path (numpy backend, Flat, a texture
        the kernel can't crop or a blend it can't compose). heights: a pooled
        batch's shared _heights_snapshot()."""
        key = (mip, cx, cy)
        if key in self._cache:
            return None
        prepared = self._prepare_rect(mip, *self._chunk_rect(mip, cx, cy), heights=heights)
        if prepared is None:
            return None
        kernel, scratch, args, holds = prepared
        return ChunkJob(key, self._mutation_epoch, kernel, scratch, args, holds)

    def install_chunk(self, job: ChunkJob, painted: bool) -> bool:
        """Caches a finished job's pixels as get_chunk() would have. Refused
        (False) if the kernel declined, if any mutation ran since the job was
        prepared (patch() only fixes chunks already cached, never one in
        flight), or if a paint already built the chunk. Only for an uncached
        key: a pooled patch's slice of a cached chunk goes through
        install_slice()."""
        if not painted or job.epoch != self._mutation_epoch or job.key in self._cache:
            return False
        self._insert(job.key, job.scratch)
        return True

    def prepare_rect_job(
        self, mip: int, cx: int, cy: int, ix0: int, iy0: int, ix1: int, iy1: int, heights: np.ndarray | None = None
    ) -> RectJob | None:
        """A pooled patch's job for slice [ix0, ix1) x [iy0, iy1) (LEVEL
        pixels) of CACHED chunk (mip, cx, cy), everything but the kernel run
        here. None means composite the slice synchronously: the chunk is not
        cached, or the rect has no worker-safe path. heights as in
        prepare_chunk_job()."""
        key = (mip, cx, cy)
        chunk = self._cache.get(key)
        if chunk is None:
            return None
        prepared = self._prepare_rect(mip, ix0, iy0, ix1, iy1, heights=heights)
        if prepared is None:
            return None
        kernel, scratch, args, holds = prepared
        return RectJob(key, self._mutation_epoch, kernel, scratch, args, holds, chunk=chunk, slice=(ix0, iy0, ix1, iy1))

    def install_slice(self, job: RectJob, painted: bool) -> bool:
        """Writes a finished RectJob's pixels into its chunk's slice. Refused
        (False) if the kernel declined, if any mutation ran since the prepare,
        or if the key no longer holds the very array the job was prepared
        against (evicted, or replaced by a fresh composite). Never inserts,
        and never touches the LRU order: a patch is not a use."""
        if not painted or job.epoch != self._mutation_epoch or self._cache.get(job.key) is not job.chunk:
            return False
        self._write_slice(job.chunk, *job.key, *job.slice, job.scratch)
        return True

    def _write_slice(
        self, chunk: np.ndarray, mip: int, cx: int, cy: int, ix0: int, iy0: int, ix1: int, iy1: int, pixels: np.ndarray
    ) -> None:
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        chunk[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = pixels
        self._stamp((mip, cx, cy))

    def set_chunk_executor(self, executor, workers: int) -> None:
        """The pool render_rect() composites missing chunks on (TASK-031.64),
        `workers` its thread count; None or 0 keeps every chunk on the calling
        thread. executor.submit(fn) must return a concurrent.futures.Future.
        The viewer shares margin_warm's pool (margin_warm.chunk_pool())."""
        self._chunk_executor = executor if workers > 0 else None
        self._chunk_workers = workers if executor is not None else 0

    def pools_chunks(self) -> bool:
        """Whether a missing chunk composites on a worker: a pool is set and
        the cache has a job path (Stepped, Sloped and iso Flat on the native
        backend). The viewer warms the zoom-out reach only then (TASK-031.11 D1)."""
        return self._chunk_executor is not None and self._has_unit_pack and composite_backend.native is not None

    def _pools(self, keys: list[tuple[int, int, int]]) -> bool:
        """Whether render_rect() takes _pooled_chunks() for `keys`:
        pools_chunks(), and at least POOL_MIN_MISSING of them are uncached."""
        if not self.pools_chunks():
            return False
        missing = 0
        for key in keys:
            if key not in self._cache:
                missing += 1
                if missing >= self.POOL_MIN_MISSING:
                    return True
        return False

    def _heights_snapshot(self, mip: int) -> np.ndarray | None:
        """One copy of the heights the kernel reads that the GUI thread writes
        in place, for every job of a pooled batch; None when there is none."""
        return None

    def _derive_pack_over(self, mip: int, rects: list[tuple[int, int, int, int]]) -> None:
        """Derives the level's pending unit-pack tiles under `rects` before a
        pooled batch's first prepare, so no later prepare writes the pack."""

    def _derive_pack_for_slices(self, mip: int, rects: list[tuple[int, int, int, int]]) -> None:
        """A pooled patch batch's _derive_pack_over(): `rects` (LEVEL pixels)
        are scattered slices, whose bounding rect can span the map."""

    def _pooled_chunks(self, mip: int, keys: list[tuple[int, int, int]]):
        """Yields get_chunk(*key) for each of `keys` in order, the misses
        composited on the chunk pool (TASK-031.64). Per POOL_BATCH keys:

        1. Prepare each miss on this thread (prepare_chunk_job()) and submit
           its kernel at once, so prepare overlaps the kernels. Nothing is
           inserted.
        2. Join (join_chunk_jobs()): this thread runs unstarted jobs too.
        3. In key order, a painted job goes through install_chunk() and
           yields its scratch; anything else yields get_chunk(), which is a
           touch on a hit and a synchronous composite on a declined miss.

        Why that is the serial loop's result, LRU included: nothing pumps the
        event loop between the first prepare and the last install, so no
        mutation interleaves (_mutation_epoch cannot move; install_chunk()
        still checks). Workers run ChunkJob.run() alone, a nogil kernel over
        arrays the job owns or shares copy-on-write, never the cache, Qt or
        perf_trace. Pass 3 performs the serial loop's cache operations in its
        order and passes 1-2 insert nothing, so evictions match; a chunk
        evicted by a later install still paints from its scratch. The batch
        shares one heights snapshot (_heights_snapshot()) and derives the
        pack first (_derive_pack_over()), all pending tiles or the batch's
        bounding candidates, so no prepare writes the pack while a job
        may read it, and no _unshare() copy is stranded per job. The one
        in-place pack write with no copy, UnitPack._slot() on addrs/dims,
        stays safe because every slot is assigned before the first submit."""
        executor = self._chunk_executor
        for start in range(0, len(keys), self.POOL_BATCH):
            batch = keys[start : start + self.POOL_BATCH]
            missing = [i for i, key in enumerate(batch) if key not in self._cache]
            jobs: list[ChunkJob | None] = [None] * len(batch)
            painted: list[bool] = [False] * len(batch)
            if missing:
                self._derive_pack_over(mip, [self._chunk_rect(*batch[i]) for i in missing])
                snapshot = self._heights_snapshot(mip)
                submitted = []
                try:
                    for i in missing:
                        job = self.prepare_chunk_job(*batch[i], heights=snapshot)
                        if job is None:
                            continue
                        try:
                            future = executor.submit(job.run)
                        except Exception as exc:  # noqa: BLE001
                            debug_log.log(f"pooled paint: submit failed for chunk {job.key} ({exc!r})", level=debug_log.ERROR)
                            future = None
                        jobs[i] = job
                        submitted.append((i, job, future))
                except BaseException:
                    # A prepare raised: don't leave this batch's kernels queued ahead of the next paint.
                    for _i, _job, future in submitted:
                        if future is not None:
                            future.cancel()
                    raise
                del snapshot
                with perf_trace.repaint_phase("join"):
                    outcomes = join_chunk_jobs([(job, future) for _i, job, future in submitted])
                for (i, _job, _future), ok in zip(submitted, outcomes, strict=True):
                    painted[i] = ok
                del submitted
            for i, key in enumerate(batch):
                job, jobs[i] = jobs[i], None
                if job is not None and self.install_chunk(job, painted[i]):
                    yield job.scratch
                else:
                    yield self.get_chunk(*key)

    def render_rect(self, x0: int, y0: int, x1: int, y1: int, mip: int = 0) -> np.ndarray:
        """Assembles pixels for [x0, x1) x [y0, y1) -- LEVEL `mip` pixels,
        clipped to that level's own canvas bounds -- from chunks, fetching/
        compositing each via get_chunk() as needed. The stitched result
        must be byte-identical to the corresponding crop of an independent
        full render at that level regardless of chunk request order or
        what was already cached -- see tools/verify_iso_chunks.py (Stepped)
        / tests/test_flat_chunks.py (Flat) at mip=0, tests/
        test_mip_geometry.py (Phase B-D-b) at mip != 0. With a chunk pool
        set (set_chunk_executor()) the misses composite on it instead
        (_pooled_chunks()), to the same pixels and cache state."""
        canvas_w, canvas_h = self.canvas_dims(mip)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(canvas_w, x1), min(canvas_h, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        cx0, cy0 = x0 // self.chunk_px, y0 // self.chunk_px
        cx1, cy1 = (x1 - 1) // self.chunk_px, (y1 - 1) // self.chunk_px
        keys = [(mip, cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)]
        chunks = self._pooled_chunks(mip, keys) if self._pools(keys) else (self.get_chunk(*key) for key in keys)
        for (_mip, cx, cy), chunk in zip(keys, chunks, strict=True):
            chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
            ox0, oy0 = max(x0, chunk_x0), max(y0, chunk_y0)
            ox1, oy1 = min(x1, chunk_x0 + chunk.shape[1]), min(y1, chunk_y0 + chunk.shape[0])
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            out[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0] = chunk[
                oy0 - chunk_y0 : oy1 - chunk_y0, ox0 - chunk_x0 : ox1 - chunk_x0
            ]
        return out

    def _init_terrain_ids(self) -> None:
        """Builds the terrain-id and layer mirrors on either backend (2.9 ms
        each at 240x240, 11 ms at 480x480): the native composite reads the
        first, terrain blending both. A full re-render builds a fresh cache,
        so fresh mirrors, and an empty blend key map over them, which the
        composites fill tile by tile as they first read each key."""
        self._terrain_ids = render_paint.tile_field_mirror(self.scenario, "terrain_id")
        self._layer_ids = render_paint.tile_field_mirror(self.scenario, "layer")
        mm = self.scenario.map_manager
        self._blend_key_map = terrain_blend.KeyMap(mm.map_width, mm.map_height)

    def _refresh_terrain_ids(self, bbox: tuple[int, int, int, int]) -> None:
        """Re-reads the mirror over bbox (REFERENCE canvas pixels) after a
        terrain edit. Every in-app terrain write reaches the cache through
        patch() or invalidate_region() with a bbox that covers each dirty
        tile's footprint, and tiles_in_screen_rect's swept test (Flat: the
        tile grid under the rect) returns a superset of the tiles painting
        into it, so every dirty tile is re-read, layer mirror included.
        A whole-canvas bbox reads every tile directly, which beats the rect
        enumeration at that size.

        The invariant the blend key map rests on: this is the only in-place
        writer of either mirror, and it invalidates the map's keys for
        exactly the tiles whose terrain id or layer it changed (and their 8
        neighbours). By value, not by bbox, so a re-read that changes nothing
        (an elevation-only edit, refresh_wholesale()) keeps every key. Any
        other mirror write must go through here or invalidate the map itself;
        replacing a mirror array is safe, since the map is bound to both. An
        invalidated key may come back from the map's stash, but only for a
        tile whose own and neighbours' values are exactly what it had then."""
        if self._terrain_ids is None:
            return
        px0, py0, px1, py1 = bbox
        if px1 <= px0 or py1 <= py0:
            return
        mm = self.scenario.map_manager
        w, h = mm.map_width, mm.map_height
        canvas_w, canvas_h = self.canvas_dims(0)
        if px0 <= 0 and py0 <= 0 and px1 >= canvas_w and py1 >= canvas_h:
            idx = np.arange(w * h)
        elif self.style == "flat":
            t = self.tile_px
            tx0, ty0 = max(px0 // t, 0), max(py0 // t, 0)
            tx1, ty1 = min(-(-px1 // t), w), min(-(-py1 // t), h)
            ys, xs = np.mgrid[ty0:ty1, tx0:tx1]
            idx = (ys * w + xs).ravel()
        else:
            tiles = iso_geometry.tiles_in_screen_rect(px0, py0, px1, py1, w, h, self.proj)
            idx = tiles[:, 1] * w + tiles[:, 0]
        if idx.size:
            tids = native_composite._terrain_ids(self.scenario, idx)
            layers = render_paint.tile_field_mirror(self.scenario, "layer", idx)
            changed = idx[(self._terrain_ids[idx] != tids) | (self._layer_ids[idx] != layers)]
            self._terrain_ids[idx] = tids
            self._layer_ids[idx] = layers
            if self._blend_key_map is not None:
                self._blend_key_map.invalidate(changed)

    def _refresh_unit_sources(self) -> None:
        """Rebuilds whatever per-cache structure holds units, after the unit
        filter changed. Overridden by FlatChunkCache, whose
        _refresh_source_caches() is a deliberate no-op once unit_draws
        exists."""
        self._mutation_epoch += 1
        self._record_flag_hosts()
        self._record_wall_verdict()
        self._refresh_source_caches()

    def set_unit_filter(self, unit_filter: UnitFilter) -> None:
        """Swaps the unit filter and makes it actually visible -- phase 3's
        P3-a.

        Three steps, and skipping the LAST one is the trap this method
        exists to close: units are composited into cached chunk PIXELS, so
        storing the filter and rebuilding the source structures alone leaves
        every already-composited chunk showing the old set of units. The
        filter would appear to do nothing until the user happened to scroll
        somewhere uncached, which reads as "the toggle is broken" rather
        than "the cache is stale".

        A no-op guard on an unchanged filter is deliberate rather than
        missing: UnitFilter is frozen and compares by value, and evicting
        the whole canvas is the single most expensive thing this class can
        be asked to do."""
        if unit_filter == self.unit_filter:
            return
        self.unit_filter = unit_filter
        self._refresh_unit_sources()
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def set_layers(self, layers: LayerState) -> None:
        """Swaps the View > Layers state and makes it actually visible.

        Same three-step shape as set_unit_filter() above, including its
        no-op guard (LayerState is frozen and compares by value, and
        evicting the whole canvas is the most expensive thing this class
        can be asked to do), and the same final step is the trap: both
        layers are baked into cached chunk PIXELS, so storing the state
        alone leaves every already-composited chunk showing the old render.

        The kinds of field cost different things, so they are diffed rather
        than all paying for the worst case:

        - A DRAW-ONLY field (_DRAW_ONLY_LAYER_FIELDS: colour_mood) is in no
          chunk at all, since MapCanvasItem paints it over whatever a chunk
          holds, so storing it is the whole update and nothing is evicted.
        - A PAINT-TIME field (terrain_textures, terrain_blend) is read straight off
          self.layers by _composite_rect, so eviction alone is the whole
          update.
        - A BUILD-TIME field (_BUILD_TIME_LAYER_FIELDS: farm_overlay,
          small_trees, hero_glow, and Translucent Walls through wall_alpha) is
          baked into the SpriteLayer's own contents when it is BUILT, so it
          needs _refresh_unit_sources() first. Skipping that would leave
          _LeveledChunkCache._level()'s `gen != self._source_gen`
          short-circuit holding and the level serving the pre-flip layer --
          the under-repaint class set_sprites_enabled() documents, not
          cosmetic staleness.

        FlatChunkCache inherits this unchanged and correctly: it reads
        farm_overlay at build time (its drape and FLAT_DRAPED icons, GH #150),
        rebuilt by the same _refresh_unit_sources(), and reaches none of the
        other build-time fields (icon_for() contain-fits every unit into its
        own footprint), so those flips repaint to the same pixels.

        A scene-overlay row (Fish Outlines) never reaches a chunk: render_key()
        resets it, so toggling one is the equal-state no-op (TASK-499)."""
        layers = render_key(layers)
        if layers == self.layers:
            return
        if layers == replace(self.layers, **{f: getattr(layers, f) for f in _DRAW_ONLY_LAYER_FIELDS}):
            self.layers = layers
            return
        rebuild = any(
            getattr(layers, f) != getattr(self.layers, f) for f in _BUILD_TIME_LAYER_FIELDS
        )
        self.layers = layers
        if rebuild:
            self._refresh_unit_sources()
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def set_grid(self, grid: GridBake) -> None:
        """Swaps the baked View > Grid spec and makes it visible: set_layers()'
        shape for a paint-time field, so eviction alone is the whole update.
        Stored rather than passed per call so patch()/patch_rects() reach it
        too; a per-call argument would punch grid-free rects into every edit."""
        if grid == self.grid:
            return
        self.grid = grid
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def set_sprites_enabled(self, enabled: bool) -> None:
        """Stores whether this cache composites real .sld sprites -- P3-g's
        toggle. The base implementation ONLY stores the value; every concrete
        subclass now overrides it to actually rebuild and evict
        (_LeveledChunkCache for Stepped and Sloped, FlatChunkCache from P3-g7),
        so this body is reached only by a subclass that has yet to grow a
        sprite path. Storing the value here regardless is what lets
        ViewerWindow carry the toggle's state across an Elevation View switch
        without special-casing which style is live.

        Deliberately on the BASE class rather than on IsoChunkCache: it is the
        hook each style's sprite path fills in by overriding."""
        self.sprites_enabled = enabled

    def farms_draped(self) -> bool:
        """Whether this cache drapes farms as terrain, in every style: units
        shown, sprites on and the Farm Overlay layer (GH #150, TASK-015)."""
        return self.with_units and self.sprites_enabled and self.layers.farm_overlay
