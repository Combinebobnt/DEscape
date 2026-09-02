"""Qt-free LRU caches of composited canvas chunks, one per terrain style.

Split out of render.py, which had grown past 3700 lines. The dependency runs
one way only: this module imports render, render.py never imports it back --
every ChunkCache mention still in render.py is a comment or a docstring, not
one is code, so the cluster is an acyclic leaf. A re-export shim would have
manufactured a cycle that only survived by deferring every dereference to
call time, which is why the split landed as one commit rather than two.

Back-references reach render through ATTRIBUTE ACCESS (render.foo(...)),
never `from descape.render import foo`. That is load-bearing, not style:
tests/test_mip_geometry.py monkeypatches render._building_bboxes_iso, which
IsoChunkCache and SlopedChunkCache call, and a from-import binds at import
time and would turn that patch into a silent no-op. All 13 back-references
route through `render.` so a future patch target cannot break either.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from descape import iso_geometry, render, settings
from descape.scenario_io import LoadedScenario
from descape.unit_filter import UnitFilter


# Default chunk size for IsoChunkCache -- measured, not guessed: benched
# CHUNK_PX in {256, 512, 1024} via tools/verify_iso_chunks.py on this
# project's largest real map (480x480, tile_px=32). 1024 wins cold
# full-canvas assembly by only ~8% over 512 (4.7s vs 5.1s) while being a
# much coarser invalidation granularity for a moving viewport; 256 loses on
# both cold assembly (~6s) and warm single-tile patch cost (~44ms vs ~28ms).
DEFAULT_CHUNK_PX = 512


class _ChunkCacheBase:
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
      - a `style` class attribute ("stepped" / "flat") -- checked at the
        MapView.set_source() boundary (Phase B-E) to catch a cache wired to
        the wrong terrain style at construction time, rather than only once
        an edit exposes the mismatch later.

    Coordinate-space convention (Phase B-D-a; PLAN_MIPS.md never states
    this explicitly): render_rect()/get_chunk()/canvas_dims() all take
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

    def _init_mip_levels(self, tile_px_by_level: dict[int, int]) -> None:
        """Enumerates the level set ONCE, at construction -- never lazily.
        Level 0 must be present and must equal self.tile_px (D2: scene
        space is pinned to the reference level, permanently).

        Phase B-D-a passes a literal {0: self.tile_px} (no other levels
        exist yet); Phase B-D-b replaces that call with a real per-level
        set from iso_geometry.mip_projections_for()/mip_tile_px_candidates().
        Enumerating eagerly (geometry only -- tile_px/proj are cheap) while
        leaving PIXELS and per-level source state lazy is what phase b's
        non-tautology byte-identity test depends on: the level set must
        already be fixed before that test's ground-truth call gets its
        tile_pixels_for_map() monkeypatched.

        Level index L means tile_px = reference_tile_px * 2**L: POSITIVE L
        is FINER (mip-up), NEGATIVE L is COARSER (mip-down) -- the only
        reading consistent with mip_for_scale()'s
        clamp(ceil(log2(scale)), ...) rule, since zooming in raises scale
        and must raise the selected level."""
        assert tile_px_by_level.get(0) == self.tile_px, (
            f"level 0 must be the reference tile_px ({self.tile_px}), got {tile_px_by_level.get(0)!r}"
        )
        self._mip_tile_px: dict[int, int] = dict(sorted(tile_px_by_level.items()))

    def mip_levels(self) -> list[int]:
        """Every enumerated level index, ascending. Always contains 0.
        Length 1 is a normal case, not a degenerate one -- e.g. a reference
        tile_px of 16 at elev_step_pct=10 has no exact neighbor at all
        (measured, see iso_geometry.mip_projections_for's own docstring)."""
        return list(self._mip_tile_px)

    def mip_tile_px(self, mip: int = 0) -> int:
        return self._mip_tile_px[mip]

    def mip_scale(self, mip: int = 0) -> float:
        """Scene-space scale factor for this level: reference_tile_px /
        level_tile_px == 2**-mip. Both operands are always powers of two in
        [MIP_MIN_TILE_PIXELS, MIP_MAX_TILE_PIXELS], so this division is
        exactly representable in binary float -- mip_scale(0) == 1.0 is an
        EXACT comparison, which is what Phase B-D-c's "keep the point
        overload at S == 1.0" safety branch relies on."""
        return self._mip_tile_px[0] / self._mip_tile_px[mip]

    def mip_for_scale(self, scale: float) -> int:
        """The coarsest level whose own scene-to-device magnification stays
        <= 1 (never magnify past what's actually resident): clamp(ceil(
        log2(scale)), min(mip_levels()), max(mip_levels())). Residual
        magnification lands in (0.5, 1.0] whenever the derived level is
        actually available -- exactly 1.0 at a power-of-two scale (no
        resampling at all), which is what keeps a unit device transform on
        the reference level (see mip_scale()'s own docstring). scale <= 0
        is guarded by returning the coarsest available level rather than
        raising -- a defensive floor for a degenerate transform, not a
        case Phase B-D-c's real callers are expected to hit.

        2026-08-28: ceil, not the naively-symmetric floor(log2(scale)) + 1
        -- they agree everywhere except exact powers of two, where ceil
        picks the coarser (better) of the two and keeps scale=1.0 selecting
        level 0 exactly. That exactness is load-bearing: testkit.
        qt_capture.scene_rect_to_array's whole 1:1 byte-identity contract
        depends on a unit device transform always landing on the reference
        level, and floor(...) + 1 would break it."""
        levels = self.mip_levels()
        if scale <= 0:
            return levels[0]
        raw = math.ceil(math.log2(scale))
        return max(levels[0], min(levels[-1], raw))

    def _bbox_to_level(self, mip: int, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """A REFERENCE-canvas-pixel bbox converted to level `mip` pixels.
        All-integer and superset-safe in BOTH directions: floor the low
        edge, ceil the high edge, so the level rect is never a strict
        subset of the true footprint (a subset would leave stale pixels
        behind). Exact because canvas dims scale by exactly
        level_tile_px/reference_tile_px -- proven per level by
        is_exact_mip (Phase B-D-b), asserted on canvas_dims() itself
        (the value the blit actually trusts) at construction.

        Identity whenever mip's tile_px equals the reference's -- true for
        every mip in Phase B-D-a, since the level set is still {0}."""
        t_ref, t_lvl = self._mip_tile_px[0], self._mip_tile_px[mip]
        if t_lvl == t_ref:
            return bbox
        px0, py0, px1, py1 = bbox
        return (
            (px0 * t_lvl) // t_ref,
            (py0 * t_lvl) // t_ref,
            -((-px1 * t_lvl) // t_ref),
            -((-py1 * t_lvl) // t_ref),
        )

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

    def _refresh_source_caches(self, elevation_changed: set | None = None) -> None:
        raise NotImplementedError

    def invalidate_units(self) -> None:
        """Forces this cache's unit-derived structures to rebuild on the next
        composite -- the explicit hook for a unit-editing mutation (place/
        move/delete/reassign), which changes unit.x/y/player without going
        through set_unit_filter()'s own unchanged-filter guard.

        The base implementation is exactly _refresh_source_caches()'s own
        unconditional (elevation_changed=None) rebuild, which is already
        correct for Iso/Sloped -- see each class's own docstring.
        FlatChunkCache overrides this: its _refresh_source_caches() is a
        no-op once unit_draws exists, so it needs the extra del/clear step
        this base version doesn't have to do.
        """
        self._refresh_source_caches()

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

        canvas_w, canvas_h = self.canvas_dims(mip)
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        x1, y1 = min(x0 + self.chunk_px, canvas_w), min(y0 + self.chunk_px, canvas_h)
        chunk = self._composite_rect(mip, x0, y0, x1, y1)
        self._cache[key] = chunk
        self._cache_bytes += chunk.nbytes
        self._cache.move_to_end(key)
        self._evict()
        return chunk

    def render_rect(self, x0: int, y0: int, x1: int, y1: int, mip: int = 0) -> np.ndarray:
        """Assembles pixels for [x0, x1) x [y0, y1) -- LEVEL `mip` pixels,
        clipped to that level's own canvas bounds -- from chunks, fetching/
        compositing each via get_chunk() as needed. The stitched result
        must be byte-identical to the corresponding crop of an independent
        full render at that level regardless of chunk request order or
        what was already cached -- see tools/verify_iso_chunks.py (Stepped)
        / tests/test_flat_chunks.py (Flat) at mip=0, tests/
        test_mip_geometry.py (Phase B-D-b) at mip != 0."""
        canvas_w, canvas_h = self.canvas_dims(mip)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(canvas_w, x1), min(canvas_h, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        cx0, cy0 = x0 // self.chunk_px, y0 // self.chunk_px
        cx1, cy1 = (x1 - 1) // self.chunk_px, (y1 - 1) // self.chunk_px
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                chunk = self.get_chunk(mip, cx, cy)
                chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
                ox0, oy0 = max(x0, chunk_x0), max(y0, chunk_y0)
                ox1, oy1 = min(x1, chunk_x0 + chunk.shape[1]), min(y1, chunk_y0 + chunk.shape[0])
                if ox1 <= ox0 or oy1 <= oy0:
                    continue
                out[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0] = chunk[
                    oy0 - chunk_y0 : oy1 - chunk_y0, ox0 - chunk_x0 : ox1 - chunk_x0
                ]
        return out

    def patch(
        self, bbox: tuple[int, int, int, int], elevation_changed: set | None = None
    ) -> None:
        """"Patch, don't drop": for every chunk CURRENTLY cached, at every
        RESIDENT mip level, that bbox (REFERENCE canvas pixels) overlaps
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
        first (self._refresh_source_caches()) -- see each subclass's own
        docstring for what that means to it.

        elevation_changed (draw-perf plan Step 3): the subset of this edit's
        dirty tiles whose elevation actually moved (see
        render._dirty_screen_bbox()'s own docstring for where this comes
        from) -- passed straight through to _refresh_source_caches(). None
        (the default) means "unknown", which every _refresh_source_caches()
        override treats as "assume the worst and rebuild everything", i.e.
        today's behavior -- every call site that doesn't yet compute this
        set keeps its exact prior cost and correctness. The degenerate-bbox
        check runs BEFORE this call now (it used to run after, paying a full
        rebuild for a bbox with nothing to patch)."""
        px0, py0, px1, py1 = bbox
        if px1 <= px0 or py1 <= py0:
            return
        self._refresh_source_caches(elevation_changed)
        for mip in sorted({key[0] for key in self._cache}):
            lx0, ly0, lx1, ly1 = self._bbox_to_level(mip, bbox)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            cx0, cy0 = lx0 // self.chunk_px, ly0 // self.chunk_px
            cx1, cy1 = (lx1 - 1) // self.chunk_px, (ly1 - 1) // self.chunk_px
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    chunk = self._cache.get((mip, cx, cy))
                    if chunk is None:
                        continue
                    chunk_x0, chunk_y0 = cx * self.chunk_px, cy * self.chunk_px
                    chunk_x1, chunk_y1 = chunk_x0 + chunk.shape[1], chunk_y0 + chunk.shape[0]
                    ix0, iy0 = max(lx0, chunk_x0), max(ly0, chunk_y0)
                    ix1, iy1 = min(lx1, chunk_x1), min(ly1, chunk_y1)
                    if ix1 <= ix0 or iy1 <= iy0:
                        continue
                    patched = self._composite_rect(mip, ix0, iy0, ix1, iy1)
                    chunk[iy0 - chunk_y0 : iy1 - chunk_y0, ix0 - chunk_x0 : ix1 - chunk_x0] = patched

    def patch_rects(self, rects) -> None:
        """patch() for each rect in rects -- Phase B-E's Flat edits patch
        per-tile rects rather than one union bbox (a union over a scattered
        undo set can span the whole map, turning patch() into a full
        recomposite; see ViewerWindow._apply_dirty's Flat branch).

        No elevation_changed parameter: this is Flat's only patch_rects()
        caller today, and FlatChunkCache._refresh_source_caches() is a
        documented no-op regardless of what's passed to it -- there is
        nothing here for that set to gate."""
        for rect in rects:
            self.patch(rect)

    def invalidate_region(self, bbox: tuple[int, int, int, int]) -> None:
        """Evicts every cached chunk, at every RESIDENT mip level, whose
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
        range instead of reusing the reference's."""
        px0, py0, px1, py1 = bbox
        ranges: dict[int, tuple[int, int, int, int]] = {}
        for mip in {key[0] for key in self._cache}:
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

    def _refresh_unit_sources(self) -> None:
        """Rebuilds whatever per-cache structure holds units, after the unit
        filter changed. Overridden by FlatChunkCache, whose
        _refresh_source_caches() is a deliberate no-op once unit_draws
        exists."""
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
        self.invalidate_region((0, 0, *self.canvas_dims(0)))

    def set_sprites_enabled(self, enabled: bool) -> None:
        """Stores whether this cache composites real .sld sprites -- P3-g's
        toggle. The base implementation ONLY stores the value; SlopedChunkCache
        overrides it to actually rebuild and evict (Track P3-g6). Flat still
        has no sprite compositor at all (composite_rect_flat takes no sprite
        argument, its own item is P3-g7), so this base no-op is still correct
        there. Storing the value here regardless is what lets ViewerWindow
        carry the toggle's state across an Elevation View switch without
        special-casing which style is live.

        This is deliberately on the BASE class, not on IsoChunkCache: it is
        the hook Sloped and (eventually) Flat's sprite paths fill in by
        overriding, the same way IsoChunkCache already does below."""
        self.sprites_enabled = enabled


# P3-g3: whether IsoChunkCache composites real .sld sprites instead of
# coloured marks. OFF for now, deliberately -- P3-g4 is the measurement gate
# that decided an opt-in toggle over default-on, and defaulting off here means
# every existing byte-identity test keeps its meaning untouched while the new
# sprite tests opt in explicitly. The headless full-render path has its own
# with_sprites argument instead, so a test can render ground truth without
# touching global state.
#
# P3-g: this is the CONSTRUCTION DEFAULT for IsoChunkCache's `sprites`
# parameter, not a live switch. Nothing reads it at call time anymore -- the
# live flag is per-cache (_ChunkCacheBase.sprites_enabled), because a
# per-window toggle cannot be a process global: module state outlives a
# window's close(), so a GUI toggle would leak across windows and into other
# tests. Defined HERE, above IsoChunkCache, rather than beside SpriteLayer
# below, purely because a default argument is evaluated when the class body
# executes -- from below the class it would raise NameError at import.
SPRITES_ENABLED = False


@dataclass
class _IsoLevel:
    """One mip level's own state (Phase B-D). tile_px/proj are enumerated
    at construction (see _ChunkCacheBase._init_mip_levels) and are pure
    geometry -- cheap, and IsoChunkCache.canvas_dims() reads .proj
    DIRECTLY, never through _level(), so sizing the cache can never trigger
    a bbox build. building_bboxes is the expensive, PROJECTION-DEPENDENT
    part (see _building_bboxes_iso/_unit_screen_bbox_iso): built lazily on
    first composite at this level, and rebuilt lazily whenever `gen` falls
    behind the cache's own _source_gen -- see IsoChunkCache._level()."""

    tile_px: int
    proj: iso_geometry.IsoProjection
    building_bboxes: dict | None = None
    gen: int = -1
    # P3-g3. Per-level for the same reason building_bboxes is: a sprite is
    # scaled by 2*half_w/NATIVE_TILE_W, so it is projection-dependent, and it
    # is built in the same lazy step rather than a second one.
    sprites: render.SpriteLayer | None = None


class IsoChunkCache(_ChunkCacheBase):
    """Qt-free LRU cache of composited Stepped-mode canvas chunks, keyed by
    (mip, chunk_x, chunk_y) -- Phase B-B of Track B.
    chunk_x/chunk_y are chunk-GRID indices: canvas pixel
    (chunk_x*chunk_px, chunk_y*chunk_px) is that chunk's own origin. mip is
    real as of Phase B-D-a (previously always 0); Phase B-D-b (this
    version) enumerates the REAL per-level projection set via
    iso_geometry.mip_projections_for(), keyed off settings.get_elev_step_pct()
    -- the same function every real proj-construction site in this module
    already calls, so the cache reading it too reproduces exactly what
    built `proj`. Neither B-D-a nor B-D-b changes a single rendered pixel
    reachable from the running app: nothing outside this class and its
    tests ever asks for mip != 0 until Phase B-D-c wires MapCanvasItem.
    paint() to a real LOD signal.

    Each chunk is composited independently via composite_rect_iso() -- the
    SAME function render_terrain_iso_with_proj()'s full loop and
    refresh_region_iso()'s per-edit patch both reduce to -- so a chunk's
    pixels never depend on which OTHER chunks happen to be cached or in
    what order they were requested (this class's own load-bearing
    correctness bar, see tools/verify_iso_chunks.py). Grid/LRU mechanics
    (get_chunk/render_rect/patch/invalidate_region) live in _ChunkCacheBase,
    shared with Phase B-E's FlatChunkCache -- this class supplies only the
    Stepped-specific pieces: canvas_dims(), _composite_rect(), and
    _refresh_source_caches().

    Wired into viewer.py's Stepped mode as of Phase B-C, via MapCanvasItem;
    also exercised standalone by tools/verify_iso_chunks.py and its own
    bench.

    max_chunks defaults to covering the WHOLE canvas at chunk_px (see
    _ChunkCacheBase._init_max_chunks()'s own docstring for why).

    NOT free memory-wise: a fully-warmed cache holds roughly as many total
    pixel bytes as the old single canvas did (this project's biggest real
    map, 480x480 at chunk_px=512, needs 480 chunks), and render_rect() (see
    below) additionally allocates a fresh full-canvas-sized stitched output
    array on every full-viewport call -- so a full-viewport composite
    transiently needs the persistent cache AND that scratch buffer at once.
    Measured on that map: peak RSS for an open+warm+edit workflow went
    743MB -> 1127MB versus the pre-B-C single-buffer approach -- a real,
    accepted cost of avoiding thrashing at the default zoom, not a memory
    win. Pass an explicit
    smaller max_chunks (as tools/verify_iso_chunks.py's eviction tests do)
    to exercise real LRU eviction instead."""

    style = "stepped"

    def __init__(
        self,
        scenario: LoadedScenario,
        elevations: np.ndarray,
        proj: iso_geometry.IsoProjection,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
        unit_filter: UnitFilter = UnitFilter(),
        sprites: bool = SPRITES_ENABLED,
    ):
        self.scenario = scenario
        self.elevations = elevations
        self.proj = proj
        self.tile_px = tile_px
        self.with_units = with_units
        self.unit_filter = unit_filter
        # Must be set before _refresh_source_caches() below, since _level()
        # reads it to decide whether to build a level's sprite layer at all.
        self.sprites_enabled = sprites
        # Phase B-D-b: the REAL per-level projection set. settings.
        # get_elev_step_pct() is read here rather than threaded through as
        # a parameter because it's exactly the same function every real
        # proj-construction site in this module already calls to build
        # `proj` itself -- so this reproduces what built `proj`, and
        # mip_projections_for's own identity-level self-check (an assert)
        # fails loudly if the two ever disagreed. Must happen before
        # _init_max_chunks(), which reads canvas_dims() -> self._levels[0].proj.
        mm = scenario.map_manager
        projs = iso_geometry.mip_projections_for(mm.map_width, mm.map_height, proj, settings.get_elev_step_pct())
        self._levels: dict[int, _IsoLevel] = {level: _IsoLevel(tile_px=p.tile_px, proj=p) for level, p in projs.items()}
        self._init_mip_levels({level: lvl.tile_px for level, lvl in self._levels.items()})
        # canvas_dims()'s own exactness assert: is_exact_mip() proves every
        # IsoProjection field scales exactly, but canvas_dims() adds the
        # skirt-headroom term (_canvas_pixel_dims) on top of proj.canvas_h --
        # the value the blit actually trusts -- so assert it here too,
        # against the real function rather than duplicating its formula
        # into iso_geometry (a second place that formula could drift).
        ref_w, ref_h = render._canvas_pixel_dims(proj)
        for lvl in self._levels.values():
            lw, lh = render._canvas_pixel_dims(lvl.proj)
            assert lw * proj.tile_px == ref_w * lvl.tile_px and lh * proj.tile_px == ref_h * lvl.tile_px, (
                f"level tile_px={lvl.tile_px}'s canvas_dims (skirt headroom included) isn't an "
                f"exact mip of the reference's -- {(lw, lh)} vs reference {(ref_w, ref_h)}"
            )
        self._source_gen = 0
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self, elevation_changed: set | None = None) -> None:
        """Rebuilds the SHARED, tile-space units_by_tile and bumps the
        source generation counter -- must run whenever elevations could
        have changed underneath this cache (construction, and every
        patch()). Deliberately does NOT rebuild any level's
        building_bboxes here: those are PROJECTION-dependent (a building's
        screen bbox depends on its center tile's elevation via
        _unit_screen_bbox_iso), so under mips they are per-level, and
        rebuilding every ENUMERATED level here would make a single edit
        pay ~15-20ms x N levels on an 11k-unit map -- for levels that may
        hold no cached chunks at all. Each level's bboxes are instead
        rebuilt lazily, in _level(), the first time that level is actually
        composited after this bump -- see _level()'s own docstring.

        elevation_changed (draw-perf plan Step 3, 3a/3b): None means
        "unknown" -- construction, or a unit-filter/sprite-toggle refresh
        via _refresh_unit_sources() -- and gets today's unconditional
        wholesale behavior (units_by_tile rebuilds, gen always bumps),
        since THOSE changed and every level's bboxes/sprites must go stale.

        A patch() caller instead passes the elevation-changed subset of its
        own edit (empty for a terrain-paint-only edit). Two changes from
        the unconditional form:

        - units_by_tile is NEVER rebuilt here (3a): no terrain or elevation
          edit can move a unit, so this dict -- a pure function of unit
          positions and the filter -- cannot go stale from a patch(), only
          from the wholesale path above.
        - the gen bump is gated (3b) on whether the edit touched any unit's
          OWN tile, using self.units_by_tile's own keys as that test:
          every unit is bucketed under its own tile among its (possibly
          several) footprint-tile buckets, so this can only ever be a
          superset of "just the own tile" -- over-triggering the bump on a
          multi-tile building's other footprint tiles, never under-
          triggering it. That matters because _level()'s gen-gated rebuild
          covers TWO things with different sensitivities (building_bboxes,
          own-tile-only via _unit_screen_bbox_iso's span>1 gate; sprite_
          draws_by_anchor, EVERY unit's own tile, 1x1 included) -- an
          under-gate here would leave a 1x1 unit's sprite anchored at a
          stale elevation with sprites on."""
        if elevation_changed is None:
            self.units_by_tile = render._units_by_tile(self.scenario, self.unit_filter) if self.with_units else {}
            self._source_gen += 1
            return
        if self.with_units and elevation_changed & self.units_by_tile.keys():
            self._source_gen += 1

    def _level(self, mip: int) -> _IsoLevel:
        """The mip level's own state, rebuilding its building_bboxes if
        stale (gen != self._source_gen). Called only from _composite_rect,
        so a level is never rebuilt just because it's resident -- only
        when actually composited. Combined with patch()'s "iterate resident
        levels only" (_ChunkCacheBase.patch), an edit at a fixed zoom (one
        level resident) pays exactly one _building_bboxes_iso rebuild,
        identical to pre-mip behavior; immediately after a mip switch (two
        levels resident), the first edit pays two -- units_by_tile itself
        is still built once per edit regardless of level count, so the
        real cost is bounded by resident level count, not enumerated level
        count."""
        lvl = self._levels[mip]
        if lvl.gen != self._source_gen:
            mm = self.scenario.map_manager
            bboxes = (
                render._building_bboxes_iso(self.units_by_tile, mm.map_width, mm.map_height, lvl.proj, self.elevations)
                if self.with_units
                else {}
            )
            # P3-g3: resolving and decoding sprites is far too expensive to do
            # per chunk, so it rides this same per-level lazy rebuild.
            # merge_sprite_bboxes is what makes composite_rect_iso pull a
            # sprite's anchor tile in as a bystander -- without it a sprite
            # clips at its owning chunk's edge.
            lvl.sprites = (
                render.sprite_draws_by_anchor(self.scenario, lvl.proj, self.elevations, self.unit_filter)
                if self.with_units and self.sprites_enabled
                else None
            )
            if lvl.sprites is not None:
                bboxes = render.merge_sprite_bboxes(bboxes, lvl.sprites)
            lvl.building_bboxes = bboxes
            lvl.gen = self._source_gen
        return lvl

    def set_sprites_enabled(self, enabled: bool) -> None:
        """Turns real .sld sprites on or off on a LIVE cache -- P3-g's toggle.

        Shaped like set_unit_filter() above, guard included, plus one extra
        step. The no-op guard matters for the same reason it does there:
        evicting the whole canvas is the most expensive thing this class can
        be asked to do.

        _refresh_unit_sources() is what bumps _source_gen, and skipping it
        would leave _level()'s `gen != self._source_gen` short-circuit
        holding: the level would keep serving the pre-flip sprite layer, with
        merge_sprite_bboxes' contribution still folded into
        building_bboxes. That is the under-repaint class of bug, not cosmetic
        staleness. invalidate_region() is the set_unit_filter() trap
        unchanged -- units are baked into chunk PIXELS, so without it the
        toggle appears to do nothing until the user scrolls somewhere
        uncached.

        The extra step is warming the resident levels eagerly (step 5). It
        looks redundant next to the lazy rebuild _level() already does, and
        it is not: turning sprites ON pays a 0.5-4.1s cold .sld decode
        (P3-g4's measurement, the reason the default is off), and the caller
        holds a wait cursor around THIS call. Left lazy, the cursor lifts
        before the real work starts and the window instead freezes for those
        seconds inside the next Qt paint, with a normal cursor and no
        indication anything is happening -- the exact UX P3-g4 cited.

        Step ordering is load-bearing: the resident set must be captured
        before invalidate_region() empties the cache and takes it away."""
        if enabled == self.sprites_enabled:
            return
        self.sprites_enabled = enabled
        resident = sorted({key[0] for key in self._cache}) or [0]
        self._refresh_unit_sources()
        for mip in resident:
            self._level(mip)
        self.invalidate_region((0, 0, *self.canvas_dims(0)))

    @property
    def building_bboxes(self) -> dict:
        """Level 0's building_bboxes, forcing a rebuild first if stale.
        Debugging/introspection accessor only -- composite_rect_iso() is
        always called with a specific level's own bboxes via _level(mip),
        never through this property."""
        return self._level(0).building_bboxes

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        """(width, height) in LEVEL `mip` canvas pixels, including skirt
        headroom -- see _canvas_pixel_dims(). Indexes self._levels[mip]
        DIRECTLY (never through _level()), so sizing the cache can never
        trigger a building_bboxes build."""
        return render._canvas_pixel_dims(self._levels[mip].proj)

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        lvl = self._level(mip)
        return render.composite_rect_iso(
            self.scenario,
            x0,
            y0,
            x1,
            y1,
            self.elevations,
            lvl.proj,
            lvl.tile_px,
            self.units_by_tile,
            lvl.building_bboxes,
            self.with_units,
            sprites=lvl.sprites,
        )


class FlatChunkCache(_ChunkCacheBase):
    """Flat mode's counterpart to IsoChunkCache -- Phase B-E of Track B. Same
    grid/LRU mechanics (_ChunkCacheBase), composited via composite_rect_flat()
    instead of composite_rect_iso()
    (see that function's own docstring for the correctness argument behind
    never needing refresh_units_over()'s fixed-point dirty-tile expansion
    here).

    max_chunks defaults to covering the WHOLE canvas at chunk_px, same as
    IsoChunkCache and for the same fitInView()-on-open reason -- unchanged
    here even though Flat's canvas is the LARGER of the two (675 MiB vs
    340 MiB at 480x480): capping this to something smaller would thrash
    at the app's own default zoom exactly
    the way IsoChunkCache's own docstring already argues against. Phase
    B-D's mip levels are the real fix for the fit-to-view memory cost, not
    a smaller cap here.

    _refresh_source_caches() is a documented no-op after construction:
    unlike Stepped's building bboxes (which move with their center tile's
    elevation, see IsoChunkCache._refresh_source_caches()), Flat's
    unit_draws (_flat_unit_draws()) derive only from unit.x/unit.y,
    unit_const, the owning player index, and map dimensions -- none of
    which any terrain or elevation edit touches. That makes patch() here
    genuinely cheaper than Stepped's per-edit ~15-20ms units_by_tile/
    building_bboxes rebuild, not just an equivalent no-op restated. If a
    future unit-editing feature (v3.5) ever makes unit_draws stale,
    invalidate_units() is the explicit way to force a rebuild -- don't
    "fix" this no-op into an unconditional rebuild instead, since that
    would silently reintroduce the per-edit cost this class exists to
    avoid paying for edits that were never about units at all."""

    style = "flat"

    def __init__(
        self,
        scenario: LoadedScenario,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
        unit_filter: UnitFilter = UnitFilter(),
    ):
        self.scenario = scenario
        self.tile_px = tile_px
        self.with_units = with_units
        self.unit_filter = unit_filter
        # Phase B-D-b: the real candidate ladder, UNFILTERED -- "unfiltered"
        # here means the MIP ladder, nothing to do with unit_filter. Flat has no
        # projection (canvas_dims() is a bare multiply, exact at every
        # tile_px, no elev_step term to break exactness), so every power of
        # two mip_tile_px_candidates() finds is a real exact level, unlike
        # Stepped's mip_projections_for() which must filter through a real
        # exactness check. Must precede _init_max_chunks(), which reads
        # canvas_dims().
        self._init_mip_levels(iso_geometry.mip_tile_px_candidates(tile_px))
        # Per-level unit_draws (Phase B-D, PLAN_MIPS.md's own "Flat's real
        # per-level work is unit_draws"). Level 0's draws live in
        # self.unit_draws (unchanged attribute, still read directly by
        # tests/test_flat_chunks.py); other levels are built lazily here.
        self._level_draws: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self, elevation_changed: set | None = None) -> None:
        """See this class's own docstring: a documented no-op once
        unit_draws already exists (nothing a terrain/elevation edit touches
        can make it stale), except at construction, where it must actually
        build unit_draws the first time. elevation_changed unused -- Flat
        has no elevation term at all, see this module's docstring."""
        if not hasattr(self, "unit_draws"):
            self.unit_draws = (
                render._flat_unit_draws(self.scenario, self.tile_px, self.unit_filter) if self.with_units else None
            )

    def invalidate_units(self) -> None:
        """Forces unit_draws to be rebuilt on the next patch()/construction-
        style refresh -- phase 3.5b's unit-editing UI is what calls this
        (via ViewerWindow._after_unit_mutation()), through the base class's
        own invalidate_units(), which this overrides because
        _refresh_source_caches()'s no-op would otherwise miss those edits.
        Also drops every other level's cached draws, same reasoning."""
        if hasattr(self, "unit_draws"):
            del self.unit_draws
        self._level_draws.clear()
        self._refresh_source_caches()

    def _refresh_unit_sources(self) -> None:
        """See _ChunkCacheBase._refresh_unit_sources(): the base version
        calls _refresh_source_caches(), which this class defines as a no-op
        once unit_draws exists, so a filter change has to go through
        invalidate_units() to force the real rebuild (level 0's draws AND
        every other level's)."""
        self.invalidate_units()

    def _level_unit_draws(self, mip: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-level _flat_unit_draws() -- Flat's counterpart to Stepped's
        per-level building_bboxes. No generation counter needed here,
        unlike IsoChunkCache._level(): unit_draws depends only on
        unit.x/unit.y, unit_const, owning player index and map dimensions,
        none of which any terrain/elevation edit touches -- the same
        property that makes _refresh_source_caches() a no-op after
        construction. invalidate_units() is what actually goes stale (a
        future unit-editing feature), and it already clears this dict --
        which is also why set_unit_filter() routes through invalidate_units()
        rather than rebuilding self.unit_draws alone: these per-level draws
        must honour the new filter too, or hidden units would keep painting
        at every zoom level except the one that happened to be resident."""
        if not self.with_units:
            return None
        if mip == 0:
            return self.unit_draws
        draws = self._level_draws.get(mip)
        if draws is None:
            draws = render._flat_unit_draws(self.scenario, self._mip_tile_px[mip], self.unit_filter)
            self._level_draws[mip] = draws
        return draws

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        mm = self.scenario.map_manager
        tile_px = self._mip_tile_px[mip]
        return mm.map_width * tile_px, mm.map_height * tile_px

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        return render.composite_rect_flat(
            self.scenario,
            x0,
            y0,
            x1,
            y1,
            self._mip_tile_px[mip],
            unit_draws=self._level_unit_draws(mip),
            with_units=self.with_units,
        )


MAX_PICK_PLANES = 4
"""How many Sloped chunk pick planes (SlopedChunkCache._pick_plane) stay
memoized at once. Sized for the access pattern, not by feel: a cursor is in
one chunk at a time and a drag along a chunk seam alternates between two, so
2 already removes the thrash case; 4 covers a diagonal drag through a
four-chunk corner without letting the memo grow into a second full-canvas
cache. At DEFAULT_CHUNK_PX=512 that is 512*512*4 = 1 MB per plane, 4 MB
capped -- negligible against the colour cache's whole-canvas budget."""


class SlopedChunkCache(_ChunkCacheBase):
    """Qt-free LRU cache of composited Sloped-mode canvas chunks -- Phase 6
    (docs/PLAN_V2_6.md)'s Track C3 counterpart to IsoChunkCache/
    FlatChunkCache. Same grid/LRU mechanics (_ChunkCacheBase), composited
    via composite_rect_sloped() instead of composite_rect_iso()/
    composite_rect_flat() -- a chunk's pixels never depend on which OTHER
    chunks happen to be cached or in what order they were requested, the
    same correctness bar tools/verify_iso_chunks.py/tests/
    test_flat_chunks.py hold their own compositors to (see tests/
    test_sloped_chunks.py for this class's own version of those checks).

    Single mip level (_init_mip_levels({0: tile_px})) -- deliberately, not
    an oversight: this is the same "level set is still {0}" intermediate
    state IsoChunkCache/FlatChunkCache themselves were in before Track
    B-D's real per-level ladder landed (see _ChunkCacheBase._init_mip_
    levels' own docstring), and it lands during that same track's active
    development. A genuine Sloped mip ladder is a legitimate follow-up --
    this class's bilinear warp would need its own per-level corner_rise_px
    array, mirroring IsoChunkCache's per-level building_bboxes -- kept out
    of Track C's own approved scope rather than built inline here.

    proj must carry corner_headroom_steps=1 (see sloped_elevations_and_
    proj()) -- built by the caller and passed in, matching IsoChunkCache's
    own convention of taking proj as a constructor parameter rather than
    building it itself."""

    style = "sloped"

    def __init__(
        self,
        scenario: LoadedScenario,
        elevations: np.ndarray,
        corner_rise: np.ndarray,
        proj: iso_geometry.IsoProjection,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
        unit_filter: UnitFilter = UnitFilter(),
        sprites: bool = SPRITES_ENABLED,
    ):
        self.scenario = scenario
        self.elevations = elevations
        # Superseded immediately by _refresh_source_caches()' own re-derivation
        # below, which owns this array from here on (see its docstring). Kept
        # as a parameter because every caller already holds the value and the
        # constructor stays parallel to elevations/proj, but a caller cannot
        # make it disagree with elevations by passing a mismatched pair.
        self.corner_rise = corner_rise
        self.proj = proj
        self.tile_px = tile_px
        self.with_units = with_units
        self.unit_filter = unit_filter
        # Must be set before _refresh_source_caches() below, since it reads
        # this to decide whether to build a sprite layer at all -- same
        # ordering trap IsoChunkCache.__init__ documents for itself.
        self.sprites_enabled = sprites
        self._pick_planes: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._init_mip_levels({0: tile_px})
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self, elevation_changed: set | None = None) -> None:
        """Rebuilds units_by_tile and building_bboxes -- unlike
        IsoChunkCache, no generation-counter laziness: this cache has only
        one (mip 0) level, so there is no "rebuild only when actually
        composited at THIS level" saving to make (see IsoChunkCache.
        _refresh_source_caches()'s own docstring for why that laziness
        exists there and would buy nothing here).

        building_bboxes reuses _building_bboxes_iso()/_unit_screen_bbox_
        iso() rather than a Sloped-specific reimplementation, but as of
        Track C5's Step 2 it must pass extra_top_px: Sloped's own proj is
        geometrically identical to Stepped's for the same map (see
        IsoProjection.corner_headroom_px's own comment), yet a unit there
        no longer sits at elevation * elev_step -- it sits at
        iso_geometry.unit_rise_px(), which under SLOPE_CORNER_RULE = "max"
        is never lower and can be higher. The "bystander" widening this
        feeds is an accepted coarse superset in the OVER direction only
        (composite_rect_iso()'s own docstring: "a union bbox flagging a
        tile as a bystander slightly more often than the tightest possible
        test would is a no-op extra paint, never a missed one"); under-
        covering is a missed paint, i.e. a stale pixel, so the headroom
        below is derived rather than assumed.

        `headroom` is an exact global bound, not the plan's "one
        elev_step": for every tile, how far its highest own corner sits
        above its own elevation, maxed over the map. One elev_step is what
        that evaluates to under this project's +-1-elevation-neighbour
        invariant, but a LOADED file is not required to satisfy that
        invariant, and this costs four vectorized array maxes next to the
        ~15-20ms _building_bboxes_iso() call it feeds. Zero on a flat map,
        so flat-map byte-identity is untouched.

        Sloped honours unit_filter exactly as Flat and Stepped do -- and,
        since C5, so does unit picking (unit_pick.pick_unit's sloped
        branch reads this same cache's corner_rise through
        MapView._sloped_cache).

        corner_rise is rebuilt here too (Track C4's Step 3), which is what
        makes Sloped editable at all: before this it was received once at
        construction and never refreshed, so an elevation edit repainted
        tiles against the pre-edit height field. It belongs in THIS method
        specifically, not in the patch() override above it, because this is
        the one hook whose contract is already "run whenever elevations
        could have changed underneath this cache" -- putting it in patch()
        would skip invalidate_region() and any future caller.

        Rebuilt whole rather than per-dirty-ring, and the cost is measured
        rather than assumed: 4.6ms at 480x480, this project's largest map,
        against a 25ms end-to-end sloped patch there (2026-08-21). So it is
        real but not dominant, and a ring-only update -- which the C4 plan
        offers as the alternative -- stays available as a later optimization
        rather than being needed now. On a unit-heavy map the
        _building_bboxes_iso() call below dwarfs it anyway (~15-20ms); on a
        blank one it is the largest single term. It also makes the redundant
        rebuild at construction a non-issue, and buys a real property for it:
        corner_rise can no longer disagree with self.elevations, however the
        constructor was called.

        self.sprites (Track P3-g6): built here, on the same "whenever
        elevations could have changed" cadence as corner_rise itself, since
        a sprite's anchor depends on corner_rise via unit_rise_px() -- an
        edit that leaves this stale would repaint sprites at their pre-edit
        height. with_farms=False always: Sloped has no warped-outline path
        for a farm foundation yet (see this repo's plan for why), so farms
        stay on the plain mark regardless of the sprite toggle.
        merge_sprite_bboxes() runs AFTER the headroom-widened building
        bboxes above, never replacing them -- both operations only ever
        grow a bbox (F3), so the order between them doesn't affect the
        result, but merging into the already-final dict avoids a second
        allocation.

        elevation_changed (draw-perf plan Step 3): None means "unknown" --
        construction, or a unit-filter/sprite-toggle refresh via
        _refresh_unit_sources() -- and gets the unconditional wholesale
        rebuild below (units_by_tile included), same as before this
        parameter existed.

        A patch() caller instead passes the elevation-changed subset of its
        own edit. 3a applies here too: units_by_tile never rebuilds on a
        patch(), since no terrain or elevation edit can move a unit. For
        corner_rise/building_bboxes/sprites this class uses ONE gate, not
        IsoChunkCache's narrower per-unit one -- deliberately: an empty
        elevation_changed (a terrain-paint-only edit) skips all three,
        since none of them depend on anything terrain-paint touches; a
        non-empty one rebuilds all three together, unconditionally, even if
        the changed tile is nowhere near a unit. That is coarser than
        IsoChunkCache's "any unit's own tile" test on purpose: corner_rise
        depends on EVERY changed tile, not just ones under units (see the
        corner_rise paragraph above), so it cannot share IsoChunkCache's
        narrower gate; and sprite_draws_by_anchor reads corner_rise itself
        (see the sprites paragraph above) -- rebuilding corner_rise while
        skipping sprites, or vice versa, would leave the sprite layer
        anchored against a height field that no longer matches
        self.corner_rise. One gate, all three rebuilt together, is the only
        form that cannot drift out of sync with itself."""
        if elevation_changed is not None:
            if not elevation_changed:
                return
            self.corner_rise = iso_geometry.corner_rise_px(self.elevations, self.proj, rule=render.SLOPE_CORNER_RULE)
            mm = self.scenario.map_manager
            headroom = render._unit_rise_headroom_px(self.corner_rise, self.elevations, self.proj)
            building_bboxes = (
                render._building_bboxes_iso(
                    self.units_by_tile, mm.map_width, mm.map_height, self.proj, self.elevations, headroom
                )
                if self.with_units
                else {}
            )
            self.sprites = (
                render.sprite_draws_by_anchor(
                    self.scenario, self.proj, self.elevations, self.unit_filter,
                    corner_rise=self.corner_rise, with_farms=False,
                )
                if self.with_units and self.sprites_enabled
                else None
            )
            self.building_bboxes = (
                render.merge_sprite_bboxes(building_bboxes, self.sprites)
                if self.sprites is not None
                else building_bboxes
            )
            return

        self.units_by_tile = render._units_by_tile(self.scenario, self.unit_filter) if self.with_units else {}
        self.corner_rise = iso_geometry.corner_rise_px(self.elevations, self.proj, rule=render.SLOPE_CORNER_RULE)
        mm = self.scenario.map_manager
        headroom = render._unit_rise_headroom_px(self.corner_rise, self.elevations, self.proj)
        building_bboxes = (
            render._building_bboxes_iso(
                self.units_by_tile, mm.map_width, mm.map_height, self.proj, self.elevations, headroom
            )
            if self.with_units
            else {}
        )
        self.sprites = (
            render.sprite_draws_by_anchor(
                self.scenario, self.proj, self.elevations, self.unit_filter,
                corner_rise=self.corner_rise, with_farms=False,
            )
            if self.with_units and self.sprites_enabled
            else None
        )
        self.building_bboxes = (
            render.merge_sprite_bboxes(building_bboxes, self.sprites)
            if self.sprites is not None
            else building_bboxes
        )

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        """(width, height) in canvas pixels -- proj.canvas_w/canvas_h alone,
        NOT render_terrain_sloped_with_proj()'s own padded allocation
        (that function adds Stepped's skirt_headroom formula on top,
        purely so a flat map's output is byte-identical to render_terrain_
        iso's -- see its own docstring). Sloped paints no skirts (see
        corner_rise_px's own docstring), so real content never needs that
        extra padding; get_chunk()'s existing high-edge clip already
        handles this cache's canvas being smaller than that function's.
        A pixel that rounds just past this tight boundary at the very top
        edge (corner_headroom_px's own float-rounding safety margin, see
        IsoProjection's comment) is silently dropped by _clipped_paint,
        the same accepted tradeoff that function's own docstring documents
        for its scratch-canvas call site -- not new here, not a correctness
        gap this class introduces.

        When sprites_enabled (Track P3-g6), returns render._canvas_pixel_
        dims(self.proj) instead -- the same skirt-padded height Stepped's
        own buffer uses -- rather than the tight bound above. Sloped's tight
        canvas fits terrain alone; a sprite reaches further than that (see
        _sprite_reach_px), and unlike Stepped, Sloped has no skirt margin to
        absorb the difference. This does not eliminate sprite-reach
        clipping -- Stepped doesn't either, see that function's own accepted
        upward gap -- it brings Sloped to the same tolerance Stepped
        already has, instead of a new, Sloped-only regression. set_sprites_
        enabled()'s invalidate_region() call is what makes the newly-
        visible strip actually get painted once this answer changes."""
        assert mip == 0, f"SlopedChunkCache has only mip level 0, got {mip}"
        if self.sprites_enabled:
            return render._canvas_pixel_dims(self.proj)
        return self.proj.canvas_w, self.proj.canvas_h

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        assert mip == 0, f"SlopedChunkCache has only mip level 0, got {mip}"
        return render.composite_rect_sloped(
            self.scenario,
            x0,
            y0,
            x1,
            y1,
            self.corner_rise,
            self.proj,
            self.tile_px,
            self.units_by_tile,
            self.building_bboxes,
            self.with_units,
            sprites=self.sprites,
        )

    def _pick_plane(self, cx: int, cy: int) -> np.ndarray:
        """Chunk (cx, cy)'s int32 tile-id plane, built on demand and
        memoized -- deliberately held OUTSIDE self._cache and outside
        self._cache_bytes, per the C plan's decision 7.

        The reason is memory, and it was decided against a measured cost:
        Track B-C already accepted a 743MB -> 1127MB RSS regression, and
        keeping an int32 plane resident alongside every cached RGB chunk
        would add ~33% on top of that (4 bytes/px against 3) for no
        correctness gain -- a pick plane is only ever needed for the ONE
        chunk under the cursor, where an RGB chunk is needed for every
        visible one. So this is a tiny separate LRU, not a second channel
        on the main one. Moving the mouse within a chunk is free; crossing
        a chunk boundary costs one ID-only recomposite, which the Gate 2
        benchmark (tools/bench_sloped_patch.py) measures separately from
        the colour path precisely because it is much cheaper.

        Same high-edge clip as get_chunk(), so a plane straddling the
        canvas edge is the same ragged shape its colour chunk is."""
        key = (cx, cy)
        cached = self._pick_planes.get(key)
        if cached is not None:
            self._pick_planes.move_to_end(key)
            return cached

        canvas_w, canvas_h = self.canvas_dims()
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        x1, y1 = min(x0 + self.chunk_px, canvas_w), min(y0 + self.chunk_px, canvas_h)
        plane = render.composite_ids_rect_sloped(
            self.scenario, x0, y0, x1, y1, self.corner_rise, self.proj, self.tile_px
        )
        self._pick_planes[key] = plane
        self._pick_planes.move_to_end(key)
        while len(self._pick_planes) > MAX_PICK_PLANES:
            self._pick_planes.popitem(last=False)
        return plane

    def pick_tile(self, sx: int, sy: int) -> tuple[int, int] | None:
        """The (tile_x, tile_y) whose sloped surface covers reference-canvas
        pixel (sx, sy), or None for a background pixel or one off-canvas --
        Track C4's answer to "what did the user just click on", and the
        Sloped counterpart to iso_geometry.screen_to_tile().

        Reference-canvas pixels, which is also scene space: mip decision D2
        pins scene space to the reference level permanently, and this cache
        has only that one level anyway, so viewer.py can pass a scene
        position through unconverted.

        Worth stating because it is the opposite of what the missing-feature
        history suggests: this is STRICTLY BETTER than Stepped's
        screen_to_tile(), which returns None on any skirt-face pixel (a
        documented accepted residual -- there is no exact analytic inverse
        for a skirt). Sloped paints no skirts at all, so it has no such
        hole: every pixel inside the ground outline resolves to exactly one
        tile, which tests/test_sloped_pick.py asserts exhaustively."""
        canvas_w, canvas_h = self.canvas_dims()
        if not (0 <= sx < canvas_w and 0 <= sy < canvas_h):
            return None
        cx, cy = sx // self.chunk_px, sy // self.chunk_px
        plane = self._pick_plane(cx, cy)
        tile_id = int(plane[sy - cy * self.chunk_px, sx - cx * self.chunk_px])
        if tile_id == render.PICK_ID_NONE:
            return None
        map_w = self.scenario.map_manager.map_width
        return tile_id % map_w, tile_id // map_w

    def patch(
        self, bbox: tuple[int, int, int, int], elevation_changed: set | None = None
    ) -> None:
        self._pick_planes.clear()
        super().patch(bbox, elevation_changed)

    def invalidate_region(self, bbox: tuple[int, int, int, int]) -> None:
        self._pick_planes.clear()
        super().invalidate_region(bbox)

    def set_unit_filter(self, unit_filter: UnitFilter) -> None:
        # Defensive, not load-bearing: the pick plane is terrain-only (see
        # _render_tile_sloped_ids), so no filter change can alter an id, and
        # Track C5 left it that way rather than folding units in. Kept so the
        # memo can never outlive a source-state change if that ever varies.
        self._pick_planes.clear()
        super().set_unit_filter(unit_filter)

    def set_sprites_enabled(self, enabled: bool) -> None:
        """Turns real .sld sprites on or off on a live cache -- Track P3-g6,
        IsoChunkCache.set_sprites_enabled()'s shape minus the mip loop (this
        cache has only one level).

        No per-level warm-eager step either: that step exists there to keep
        a wait cursor over the cold .sld decode instead of freezing the next
        Qt paint with no indication anything is happening. _refresh_source_
        caches() below already pays that decode eagerly and synchronously
        (it isn't gated behind a lazy _level() rebuild the way IsoChunkCache's
        is), so there is nothing left to warm.

        invalidate_region()'s bbox is read from canvas_dims() AFTER
        self.sprites_enabled flips -- load-bearing when turning sprites ON at
        a low elev_step_pct stop, where canvas_dims() itself grows (the Step
        0 fix): invalidating the OLD, smaller bbox would leave the newly-
        visible bottom strip never composited at all, not just stale."""
        if enabled == self.sprites_enabled:
            return
        self.sprites_enabled = enabled
        self._refresh_source_caches()
        self.invalidate_region((0, 0, *self.canvas_dims(0)))
