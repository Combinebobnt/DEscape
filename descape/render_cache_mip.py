"""_ChunkCacheBase's mip ladder: level selection, level warm jobs, residency and stamps, reduced chunks."""

from __future__ import annotations

import numpy as np

from descape import (
    asset_source,
    composite_backend,
    perf_trace,
)
from descape.render_cache_common import (
    _NEVER_STAMP,
    MIP_SCALE_SLACK,
    PACK_WARM_TILES,
    REDUCED_MAX_BYTES,
    LevelWarmJob,
    _Reduced,
    box_reduce2,
)


class _ChunkCacheMipMixin:
    """_ChunkCacheBase's mip ladder (render_cache.py's docstring has the split's rules)."""

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

        Level indices are contiguous and ordered by tile_px: POSITIVE L is
        FINER (mip-up), NEGATIVE L is COARSER (mip-down), since zooming in
        raises scale and must raise the selected level. An index is a rank,
        not log2 of the ratio: the iso ladders' native tier (96, TASK-599)
        sits between two octaves, so read the ratio from mip_tile_px()."""
        assert tile_px_by_level.get(0) == self.tile_px, (
            f"level 0 must be the reference tile_px ({self.tile_px}), got {tile_px_by_level.get(0)!r}"
        )
        self._mip_tile_px: dict[int, int] = dict(sorted(tile_px_by_level.items()))
        # The native tier crops wrap-padded texture copies; they live while a cache that may build it does.
        if any(asset_source.crop_pad(asset_source.terrain_texture_size(px), px) for px in self._mip_tile_px.values()):
            asset_source.hold_wrap_padded(self)

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
        level_tile_px. mip_scale(0) == 1.0 is an EXACT comparison (x / x),
        which is what Phase B-D-c's "keep the point overload at S == 1.0"
        safety branch relies on. Not always a power of two: the native tier
        (TASK-599) gives 2/3 at base 64 and 1/3 at base 32."""
        return self._mip_tile_px[0] / self._mip_tile_px[mip]

    def mip_for_scale(self, scale: float) -> int:
        """The coarsest level whose own scene-to-device magnification stays
        <= 1 (never magnify past what's actually resident): the coarsest
        level whose mip_tile_px() / reference tile_px >= scale, else the
        finest. On a power-of-two ladder that is clamp(ceil(log2(scale)),
        coarsest, finest); with the native tier (TASK-599) at base 64,
        scales in (1.0, 1.5] pick 96 and (1.5, 2.0] pick 128. Residual
        magnification lands in (0.5, 1.0] (or (2/3, 1.0] beside the native
        tier) whenever the derived level is available -- exactly 1.0 at a
        level's own ratio (no resampling at all), which is what keeps a unit
        device transform on the reference level (see mip_scale()'s own
        docstring). scale <= 0 is guarded by returning the coarsest
        available level rather than raising -- a defensive floor for a
        degenerate transform, not a case Phase B-D-c's real callers are
        expected to hit.

        2026-08-28: ceil, not the naively-symmetric floor(log2(scale)) + 1
        -- they agree everywhere except exact powers of two, where ceil
        picks the coarser (better) of the two and keeps scale=1.0 selecting
        level 0 exactly. That exactness is load-bearing: testkit.
        qt_capture.scene_rect_to_array's whole 1:1 byte-identity contract
        depends on a unit device transform always landing on the reference
        level, and floor(...) + 1 would break it. MIP_SCALE_SLACK keeps a
        view landed on a level's ratio (MapView's native sprite stop) on
        that level when the multiply leaves it an ulp above."""
        levels = self.mip_levels()
        if scale <= 0:
            return levels[0]
        t0 = self._mip_tile_px[0]
        bound = scale * (1.0 - MIP_SCALE_SLACK)
        for level in levels:
            if self._mip_tile_px[level] >= bound * t0:
                return level
        return levels[-1]

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

    def _level_bbox_to_reference(self, mip: int, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """The inverse of _bbox_to_level(): a level-`mip`-pixel bbox in
        REFERENCE canvas pixels, floor low and ceil high by t_ref/t_lvl, so
        _bbox_to_level() of the result is always a superset of `bbox`."""
        t_ref, t_lvl = self._mip_tile_px[0], self._mip_tile_px[mip]
        if t_lvl == t_ref:
            return bbox
        lx0, ly0, lx1, ly1 = bbox
        return (
            (lx0 * t_ref) // t_lvl,
            (ly0 * t_ref) // t_lvl,
            -((-lx1 * t_ref) // t_lvl),
            -((-ly1 * t_ref) // t_lvl),
        )

    def resident_levels(self) -> list[int]:
        """Every mip holding at least one cached chunk, ascending: the levels
        patch()/invalidate_region() fan out over when given no `levels`."""
        return sorted({key[0] for key in self._cache})

    def _select_levels(self, levels) -> list[int]:
        """resident_levels(), narrowed to `levels` when that is not None."""
        resident = self.resident_levels()
        if levels is None:
            return resident
        wanted = set(levels)
        return [mip for mip in resident if mip in wanted]

    def level_warm_job(self, mip: int) -> LevelWarmJob | None:
        """This level's sprite layer as a resumable warm, or None if there is
        nothing worth warming -- the entry point level_warm.LevelWarmer
        drives (2026-09-04 plan, Step 2).

        The base None is an unused conservative default ("nothing to warm"):
        every concrete class overrides it, _LeveledChunkCache (Stepped and
        Sloped) and FlatChunkCache."""
        return None

    def is_level_resident(self, mip: int) -> bool:
        """Whether mip's sprite/icon layer is already built -- the
        precondition margin_warm.MarginWarmer (2026-09-07 plan, Step A3.3)
        must check before ever calling get_chunk() on a level: get_chunk()
        on a not-yet-visited level synchronously builds the WHOLE level
        inside that call -- 0.6-4.3s (level_warm.py's own docstring), the
        exact freeze this whole subsystem exists to remove, now inside a
        margin-warm timer callback with no user action to blame it on.

        The base True is an unused default for a class with nothing lazy to
        build: every concrete class overrides it with its own staleness
        predicate, _LeveledChunkCache (Stepped and Sloped) and FlatChunkCache."""
        return True

    def paints_without_build(self, mip: int) -> bool:
        """Whether a composite at `mip` builds nothing level-wide: what a
        pre-rendered zoom frame needs (TASK-031.71). is_level_resident() on
        every class: Flat's counts a level with no icon layer to build as
        resident too (TASK-031.72)."""
        return self.is_level_resident(mip)

    def level_warmable(self, mip: int) -> bool:
        """Whether level_warm_job(mip) would return a job, without building
        one: the placeholder paint's "a warm will resolve this level" test
        (TASK-031.25). Mirrors level_warm_job(); each concrete class
        overrides both, and tests/test_placeholder_paint.py pins the pair."""
        return False

    def missing_chunks(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> int:
        """How many chunk keys under LEVEL-`mip` rect [x0, x1) x [y0, y1),
        clipped to canvas_dims(mip) as render_rect() clips, are not cached.
        Membership reads only (no LRU touch); 0 for an empty clipped rect."""
        canvas_w, canvas_h = self.canvas_dims(mip)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(canvas_w, x1), min(canvas_h, y1)
        if x1 <= x0 or y1 <= y0:
            return 0
        cx0, cy0, cx1, cy1 = self.chunk_index_range(mip, x0, y0, x1, y1)
        cache = self._cache
        return sum(
            (mip, cx, cy) not in cache for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)
        )

    def _unit_pack_of(self, mip: int, create: bool):
        """Level `mip`'s native unit pack: with create, the one a composite
        would use (built if missing or stale); without, the installed one only
        if it still matches its sources, else None. Only called when
        _has_unit_pack is True."""
        raise NotImplementedError

    def pack_warm_job(self, mip: int, after_level_warm: bool = False) -> LevelWarmJob | None:
        """The level's pending unit-pack tiles, derived a slice per step
        (maintainer plan 2026-09-27) so a first-touch chunk finds them done.
        Separate from level_warm_job(), whose None is pinned against
        is_level_resident().

        None when there is nothing to derive: no pack on this cache (numpy,
        units off, Flat); a level that isn't resident, unless
        after_level_warm says a level warm queued ahead of this job will make
        it so; or a resident level whose pack is current with nothing
        pending, which keeps the per-stroke-step re-arm from starting a timer.

        Byte-identical by construction: the walk calls the pack's own ensure()
        earlier than a composite would. Between slices it stops quietly once
        the level leaves residency or its pack is replaced or goes stale."""
        if not self._has_unit_pack or composite_backend.native is None or not self.with_units:
            return None
        # A deferring level's pack is refreshed by its flush; reading it here would force one mid-stroke.
        # after_level_warm: the flush job queued ahead of this one runs it first.
        if self._defers_patch(mip) and not after_level_warm:
            return None
        if self.is_level_resident(mip):
            pack = self._unit_pack_of(mip, create=False)
            if pack is not None and pack.pending().size == 0:
                return None
        elif not after_level_warm:
            return None
        return LevelWarmJob(gen=self._pack_warm_walk(mip), install=lambda _payload: True, kind="pack")

    def _pack_warm_walk(self, mip: int):
        if not self.is_level_resident(mip):
            return
        pack = self._unit_pack_of(mip, create=True)
        pending = pack.pending()
        for i in range(0, pending.size, PACK_WARM_TILES):
            yield
            if (
                not self.is_level_resident(mip)
                or self._defers_patch(mip)
                or self._unit_pack_of(mip, create=False) is not pack
            ):
                return
            pack.ensure(pending[i : i + PACK_WARM_TILES])

    def native_prime_job(self, mip: int, after_level_warm: bool = False) -> LevelWarmJob | None:
        """The level's native composite inputs, primed an atom per step
        (TASK-031.85), so a first chunk there finds them built. None on the
        base: top-down Flat has no native composite. _LeveledChunkCache
        (Stepped, iso Flat and Sloped) overrides it."""
        return None

    def native_primed(self, mip: int) -> bool:
        """Whether a native prime walk finished for this level; False on the
        base, which has none. _LeveledChunkCache overrides it."""
        return False

    def level_warmed(self, mip: int) -> bool:
        """Whether this level's process-global cold builds have run: a chunk
        was composited at it, or (Stepped, iso Flat and Sloped) its native
        prime finished. margin_warm's cold-first defer reads it (TASK-031.82
        D3). Per level only: a chunk with new blend keys or shapes can still
        pay some cold cost on a warmed level."""
        return mip in self._composited_mips or self.native_primed(mip)

    def has_chunk(self, mip: int, cx: int, cy: int) -> bool:
        """Whether chunk (mip, cx, cy) is already cached -- a plain
        membership read, no LRU touch (unlike get_chunk()'s move_to_end()).
        margin_warm.ring_chunks() uses this to drop already-resident chunks
        from a margin ring before it's ever queued, so
        MarginWarmer.is_active stays honest about remaining work instead of
        counting a chunk that needs no warming at all."""
        return (mip, cx, cy) in self._cache

    def level_chunk_count(self, mip: int, outside: tuple[int, int, int, int] | None = None) -> int:
        """How many of level `mip`'s chunks are cached, no LRU touch; with
        `outside` (an inclusive cx0, cy0, cx1, cy1), only those not in it."""
        if outside is None:
            return sum(1 for key in self._cache if key[0] == mip)
        x0, y0, x1, y1 = outside
        return sum(1 for m, cx, cy in self._cache if m == mip and not (x0 <= cx <= x1 and y0 <= cy <= y1))

    def touch_level(self, mip: int) -> None:
        """Moves level `mip`'s cached chunks to the LRU's warm end, in their
        order, as a read of each would: what a warm that must not evict them
        calls before it inserts (TASK-031.11 D5, the coarsest level)."""
        cache = self._cache
        for key in [key for key in cache if key[0] == mip]:
            cache.move_to_end(key)

    def chunk_stamps(self, mip: int, cx0: int, cy0: int, cx1: int, cy1: int) -> tuple | None:
        """The pixel stamp (_stamp()) of every chunk in the inclusive LEVEL-`mip`
        range, with canvas_dims(mip), or None if any is uncached. A pre-rendered
        zoom frame records it (TASK-031.71) and serves only while
        stamps_current() holds: every write to a cached chunk's pixels
        re-stamps it, and every other change to them removes it."""
        cache, stamps = self._cache, self._chunk_stamp
        keys = [(mip, cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)]
        if any(key not in cache for key in keys):
            return None
        return mip, self.canvas_dims(mip), tuple((key, stamps[key]) for key in keys)

    def stamps_current(self, recorded: tuple) -> bool:
        """Whether every chunk chunk_stamps() recorded is still cached with the
        same stamp, at the same canvas_dims(): its pixels are unchanged."""
        mip, dims, entries = recorded
        cache, stamps = self._cache, self._chunk_stamp
        return dims == self.canvas_dims(mip) and all(key in cache and stamps.get(key) == stamp for key, stamp in entries)

    def reduced_dims(self, mip: int, k: int) -> tuple[int, int]:
        """canvas_dims(mip) after k box_reduce2() halvings: ceil(dim / 2**k) per axis."""
        canvas_w, canvas_h = self.canvas_dims(mip)
        f = 1 << k
        return -(-canvas_w // f), -(-canvas_h // f)

    def render_reduced_rect(self, x0: int, y0: int, x1: int, y1: int, mip: int, k: int) -> np.ndarray:
        """render_rect()'s shape for level `mip` box-reduced k >= 1 times
        (TASK-031.70): [x0, x1) x [y0, y1) in reduced-k pixels, clipped to
        reduced_dims(mip, k). Byte-identical to box_reduce2() applied k times
        to a full render of the level, whatever was cached. Paint uses it at
        the coarsest level only, and it never enters mip_levels().

        Each reduced chunk (chunk_px square in reduced-k pixels) is cached in
        its own LRU (`_reduced`, REDUCED_MAX_BYTES), apart from max_bytes, and
        built from k-1 (render_rect() at k = 1) over the 2x rect. Validity is
        by stamps, not hooks at every delete: an entry records the stamp of
        every REAL level-`mip` chunk under it, geometrically (an uncached one
        as never valid), and serves only while each of those is still cached
        with the same stamp (_reduced_valid()). Never the k-1 entries' stamps:
        one is rebuilt only when read, so after an edit it could still be
        present and old while the real chunks under it changed."""
        assert k >= 1, f"render_reduced_rect() needs k >= 1, got {k}; k = 0 is render_rect()"
        rw, rh = self.reduced_dims(mip, k)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(rw, x1), min(rh, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        size = self.chunk_px
        for cy in range(y0 // size, (y1 - 1) // size + 1):
            for cx in range(x0 // size, (x1 - 1) // size + 1):
                chunk = self._reduced_chunk(mip, k, cx, cy)
                chunk_x0, chunk_y0 = cx * size, cy * size
                ox0, oy0 = max(x0, chunk_x0), max(y0, chunk_y0)
                ox1, oy1 = min(x1, chunk_x0 + chunk.shape[1]), min(y1, chunk_y0 + chunk.shape[0])
                out[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0] = chunk[
                    oy0 - chunk_y0 : oy1 - chunk_y0, ox0 - chunk_x0 : ox1 - chunk_x0
                ]
        return out

    def _reduced_parents(self, mip: int, k: int, cx: int, cy: int) -> list[tuple[int, int, int]]:
        """The real level-`mip` chunk keys under reduced chunk (mip, k, cx, cy),
        clipped to canvas_dims(mip): the 2**k x 2**k block, fewer at the edge."""
        canvas_w, canvas_h = self.canvas_dims(mip)
        span = self.chunk_px << k
        x0, y0 = cx * span, cy * span
        x1, y1 = min(x0 + span, canvas_w), min(y0 + span, canvas_h)
        if x1 <= x0 or y1 <= y0:
            return []
        cx0, cy0, cx1, cy1 = self.chunk_index_range(mip, x0, y0, x1, y1)
        return [(mip, pcx, pcy) for pcy in range(cy0, cy1 + 1) for pcx in range(cx0, cx1 + 1)]

    def _reduced_valid(self, entry: _Reduced, key: tuple[int, int, int, int]) -> bool:
        """Whether every real parent the entry at `key` recorded is still cached
        with the stamp it had at build time. Presence matters on its own: patch()
        skips an uncached chunk, so an evicted parent may have missed an edit."""
        if entry.dims != self.canvas_dims(key[0]):
            return False
        cache, stamps = self._cache, self._chunk_stamp
        return all(
            stamp != _NEVER_STAMP and parent in cache and stamps.get(parent) == stamp for parent, stamp in entry.parents
        )

    def _reduced_chunk(self, mip: int, k: int, cx: int, cy: int) -> np.ndarray:
        """Reduced chunk (mip, k, cx, cy): a valid entry, else built and stored.
        A hit is a use of its parents, which keeps them as resident as a paint
        at the level itself would: it move_to_end()s its real parents and the
        k-1 entries under it still in the reduced LRU."""
        key = (mip, k, cx, cy)
        entry = self._reduced.get(key)
        if entry is not None and self._reduced_valid(entry, key):
            self._reduced.move_to_end(key)
            cache = self._cache
            for parent, _stamp in entry.parents:
                if parent in cache:
                    cache.move_to_end(parent)
            if k > 1:
                for sub in ((mip, k - 1, 2 * cx + dx, 2 * cy + dy) for dy in (0, 1) for dx in (0, 1)):
                    if sub in self._reduced:
                        self._reduced.move_to_end(sub)
            return entry.pixels
        rw, rh = self.reduced_dims(mip, k)
        size = self.chunk_px
        x0, y0 = cx * size, cy * size
        x1, y1 = min(x0 + size, rw), min(y0 + size, rh)
        if k == 1:
            src = self.render_rect(2 * x0, 2 * y0, 2 * x1, 2 * y1, mip=mip)
        else:
            src = self.render_reduced_rect(2 * x0, 2 * y0, 2 * x1, 2 * y1, mip, k - 1)
        with perf_trace.repaint_phase("reduce"):
            pixels = box_reduce2(src)
        parents = tuple(
            (parent, self._chunk_stamp.get(parent, _NEVER_STAMP) if parent in self._cache else _NEVER_STAMP)
            for parent in self._reduced_parents(mip, k, cx, cy)
        )
        old = self._reduced.pop(key, None)
        if old is not None:
            self._reduced_bytes -= old.pixels.nbytes
        self._reduced[key] = _Reduced(pixels, parents, self.canvas_dims(mip))
        self._reduced_bytes += pixels.nbytes
        while self._reduced_bytes > REDUCED_MAX_BYTES and len(self._reduced) > 1:
            _victim_key, victim = self._reduced.popitem(last=False)
            self._reduced_bytes -= victim.pixels.nbytes
        return pixels
