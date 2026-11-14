"""Batch F N2: composite_rect_iso/_sloped as one native call per rect.

Python keeps everything with Python-rounding or cache semantics and hands the
kernel flat arrays: the ordered candidates (tiles_in_screen_rect plus the
bystander merge), each candidate's texture slot and grid stamp, the index
tables, a per-level UnitPack of marks, sprites and farms, and the blend
tables (_blend_tables()). The numpy path in the render modules stays the
byte-identity oracle (tests/test_native_composite.py; blended rects in
tests/test_terrain_blend_render.py).

Paint order lives in three places that must change together, with a
KERNEL_ABI bump: render_iso._paint_candidates_iso/render_sloped._paint_candidates_sloped (the reference), the
kernel's composite_iso/composite_sloped loops, and UnitPack's row order.

What is read live on every call, never cached here, so nothing can go stale
under an edit: elevations and corner_rise, the textures and blend masks
(from asset_source's own lru, held only for the call), and every factor,
LUT, shade and grid-stamp table (from the render modules' lru caches, which tests
rebind and cache_clear()).
Cached across calls: pure index geometry keyed by (tile_px, elev_step) or by a
normalized Sloped shape, the blend sample list derived from it (the texels a
blended tile's paint reads, SteppedTables/SlopedShapes.blend_samples), and the
UnitPack, which its owning cache refreshes through the same splice funnels that
update its SpriteLayer (see UnitPack).

Terrain ids are read live too, unless the caller passes `terrain_ids`: a flat
per-tile mirror its owning chunk cache keeps and refreshes in patch() and
invalidate_region(). Scattered live reads cost ~1 ms a heavy chunk in cache
misses; a gather from the mirror costs microseconds.

Blend keys are the one input that IS memoized across calls, though not here:
they arrive as a terrain_blend.RectKeys whose KeyMap the owning chunk cache
keeps beside that mirror and invalidates in the mirror's only writer
(render_cache_base._ChunkCacheBase._refresh_terrain_ids). The op and mask tables built from them
are still per call.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import repeat
from operator import attrgetter

import numpy as np

from descape import asset_source, iso_geometry, render_iso, render_paint, render_sloped, render_sprites, render_units, terrain_blend
from descape.terrain_palette import color_for_terrain_id

# Stepped op-table layout, mirrored by DEF constants in _composite_native.pyx.
OP_DIAMOND = 0
N_FIXED = 8
PER_RISE = 5
MAX_RISE = iso_geometry.MAX_ELEVATION - iso_geometry.MIN_ELEVATION

# Unit-pack rows are (kind, y, x, slot or rgb, base rise), mirrored by DEF
# constants in _composite_native.pyx.
ROW_MARK, ROW_CONFORM, ROW_SPRITE = 0, 1, 2
ROW_W = 5

# What a with_units=False cache passes as composite_rect_*'s unit_pack: take
# the whole-rect path, with nothing to draw after the terrain.
NO_UNITS = object()
_EDGE_SIDES = tuple(side for _bit, side in render_sprites._FARM_EDGE_BITS)
_EMPTY_EXT = (1, 0, 0, 0)


def _extents(dy: np.ndarray, dx: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Each op's inclusive (min y, max y, min x, max x) over the concatenated
    columns, _EMPTY_EXT for an empty op, flat int32 (TASK-031.85)."""
    ext = np.empty((start.size - 1, 4), dtype=np.int32)
    ext[:] = _EMPTY_EXT
    full = start[1:] > start[:-1]
    if full.any():
        # Empty ops share their successor's offset, so the non-empty starts alone segment the columns.
        idx = start[:-1][full]
        ext[full, 0] = np.minimum.reduceat(dy, idx)
        ext[full, 1] = np.maximum.reduceat(dy, idx)
        ext[full, 2] = np.minimum.reduceat(dx, idx)
        ext[full, 3] = np.maximum.reduceat(dx, idx)
    return ext.reshape(-1)


_I32 = np.iinfo(np.int32)


def _i32(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=np.int32)
    wide = np.concatenate(parts)
    if wide.dtype.kind not in "iu":
        if not np.array_equal(wide.astype(np.int32), wide):
            raise OverflowError("an index table value does not fit in int32")
    elif wide.size and (int(wide.min()) < _I32.min or int(wide.max()) > _I32.max):
        raise OverflowError("an index table value does not fit in int32")
    return wide.astype(np.int32)


def _blend_samples(tile_px: int, sy: list[np.ndarray], sx: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The distinct (sy, sx) across the given source columns, row-major, as
    two contiguous int32 arrays: the texels the kernel composes per blended
    tile. Built fresh, never written in place, since a worker may hold it.
    A presence bitmap over [0, tile_px²) stands in for np.unique (TASK-031.85)."""
    flat = np.concatenate([np.zeros(0, np.int64), *(y.astype(np.int64) * tile_px + x for y, x in zip(sy, sx, strict=True))])
    present = np.zeros(tile_px * tile_px, dtype=bool)
    if flat.size:
        # A negative index would wrap silently in present[flat].
        assert int(flat.min()) >= 0 and int(flat.max()) < tile_px * tile_px
        present[flat] = True
    flat = np.flatnonzero(present)
    return np.ascontiguousarray(flat // tile_px, dtype=np.int32), np.ascontiguousarray(flat % tile_px, dtype=np.int32)


def _csr(ops: list[tuple], src: bool) -> dict:
    """Concatenates per-op (dst_y, dst_x[, src_y, src_x]) into int32 columns
    with int64 start offsets and per-op inclusive extents."""
    lens = [op[0].size for op in ops]
    start = np.zeros(len(ops) + 1, dtype=np.int64)
    np.cumsum(lens, out=start[1:])
    cols = {
        "dy": _i32([op[0] for op in ops]),
        "dx": _i32([op[1] for op in ops]),
        "start": start,
    }
    cols["ext"] = _extents(cols["dy"], cols["dx"], start)
    if src:
        cols["sy"] = _i32([op[2] for op in ops])
        cols["sx"] = _i32([op[3] for op in ops])
    return cols


def _fixed_ops(tile_px: int) -> list[tuple]:
    """The Stepped op table's N_FIXED (op, factor key) pairs. The key is the
    render_iso._seam_factors/_shadow_factors key for a darken op, None for a
    scatter or solid one. Shared by SteppedTables and prime_walk()."""
    g = iso_geometry
    return [
        (g.diamond_indices(tile_px), None),
        *((g.seam_edge_indices(tile_px, side), ("seam", side)) for side in ("up_left", "up_right")),
        (g.seam_apex_indices(tile_px), ("seam", "apex")),
        *((g.tile_edge_indices(tile_px, side), None) for side in _EDGE_SIDES),
    ]


def _rise_ops(tile_px: int, rise: int) -> list[tuple]:
    """One rise's PER_RISE (op, factor key) pairs, as _fixed_ops()."""
    g = iso_geometry
    return [
        (g.skirt_quad_indices(tile_px, rise, "left"), None),
        (g.skirt_quad_indices(tile_px, rise, "right"), None),
        *((g.shadow_quad_indices(tile_px, rise, side)[:2], ("shadow", rise, side)) for side in ("up_left", "up_right")),
        (g.shadow_apex_indices(tile_px, rise)[:2], ("shadow", rise, "apex")),
    ]


class SteppedTables:
    """Every Stepped index producer's output for one (tile_px, elev_step),
    rises 1..MAX_RISE, in the kernel's op order. Geometry only: the darken
    factors are fetched per call (stepped_factors)."""

    def __init__(self, tile_px: int, elev_step: int):
        pairs = _fixed_ops(tile_px)
        assert len(pairs) == N_FIXED
        for r in range(1, MAX_RISE + 1):
            pairs += _rise_ops(tile_px, r * elev_step)
        assert len(pairs) == N_FIXED + MAX_RISE * PER_RISE
        ops = [op for op, _key in pairs]
        # factor_keys[i] is op i's factor key (_fixed_ops()).
        keys = [key for _op, key in pairs]
        # One src entry per dst entry in every op, zeros where an op samples
        # nothing, so the four columns share the start offsets.
        cols = _csr([op if len(op) == 4 else (*op, np.zeros_like(op[0]), np.zeros_like(op[0])) for op in ops], src=True)
        for name in ("sy", "sx"):
            assert cols[name].size == 0 or (cols[name].min() >= 0 and cols[name].max() < tile_px)
        self.tile_px = tile_px
        self.cols = cols
        self.factor_keys = keys
        self.op_len = np.diff(cols["start"])
        # Every op whose paint reads a Src (the kernel's _op_scatter calls):
        # the diamond and both skirts of every rise.
        start = cols["start"]
        reads = [OP_DIAMOND, *(N_FIXED + (r - 1) * PER_RISE + side for r in range(1, MAX_RISE + 1) for side in (0, 1))]
        self.blend_samples = _blend_samples(
            tile_px, [cols["sy"][start[i]:start[i + 1]] for i in reads], [cols["sx"][start[i]:start[i + 1]] for i in reads],
        )
        fstart = np.full(len(ops), -1, dtype=np.int64)
        total = 0
        for i, key in enumerate(keys):
            if key is not None:
                fstart[i] = total
                total += int(self.op_len[i])
        self.fstart = fstart
        self._factor_src: list | None = None
        self._factors: np.ndarray | None = None

    def factors(self) -> np.ndarray:
        """The darken factors, fetched from render_iso's lru caches on every call
        and re-concatenated only when one of them is a different array than
        last time (a test rebinding SEAM_SHADE then cache_clear()ing)."""
        src = []
        for key in self.factor_keys:
            if key is None:
                continue
            if key[0] == "seam":
                src.append(render_iso._seam_factors(self.tile_px, key[1]))
            else:
                src.append(render_iso._shadow_factors(self.tile_px, key[1], key[2]))
        cached = self._factor_src
        if cached is None or len(cached) != len(src) or any(a is not b for a, b in zip(cached, src, strict=True)):
            lens = [int(n) for n, key in zip(self.op_len, self.factor_keys, strict=True) if key is not None]
            for a, n in zip(src, lens, strict=True):
                if a.size != n:
                    raise ValueError("a darken factor table is not aligned with its index producer")
            self._factors = np.ascontiguousarray(np.concatenate(src), dtype=np.float32) if src else np.zeros(0, np.float32)
            self._factor_src = src
        return self._factors

    def kernel_tables(self) -> tuple:
        c = self.cols
        luts = np.ascontiguousarray(np.stack([render_iso._skirt_lut("left"), render_iso._skirt_lut("right")]), dtype=np.uint8)
        return (c["dy"], c["dx"], c["sy"], c["sx"], None, c["start"], c["ext"], self.fstart, self.factors(), luts)


@lru_cache(maxsize=16)
def stepped_tables(tile_px: int, elev_step: int) -> SteppedTables:
    return SteppedTables(tile_px, elev_step)


class SlopedShapes:
    """Sloped quad and farm-edge geometry per normalized corner shape, for one
    tile_px. Shapes accumulate as candidates reveal them; both producers are
    normalization-invariant (they subtract the corner minimum themselves), so
    a shape's arrays are the same for every tile that has it. A key is the
    four corners plus the tile's fold (TASK-008), False on every planar
    shape; the fold reaches the quad's resample only, never its edges."""

    def __init__(self, tile_px: int):
        self.tile_px = tile_px
        self.ids: dict[tuple[int, int, int, int, bool], int] = {}
        self.corners: list[tuple[int, int, int, int, bool]] = []
        self._quads: list[tuple] = []
        self._edges: list[tuple] = []
        self._packed: tuple | None = None
        diamond = iso_geometry.diamond_indices(tile_px)
        self.diamond = _pack_ops([diamond[:2]])
        self._diamond_src = diamond[2:4]

    def shape_ids(self, keys: list[tuple[int, int, int, int, bool]]) -> list[int]:
        out = []
        for key in keys:
            sid = self.ids.get(key)
            if sid is None:
                sid = len(self.corners)
                g = iso_geometry
                dy, dx, sy, sx, uv = g.sloped_quad_indices(self.tile_px, *key[:4], nwse=key[4])
                assert sy.size == 0 or (sy.min() >= 0 and sy.max() < self.tile_px)
                assert sx.size == 0 or (sx.min() >= 0 and sx.max() < self.tile_px)
                self._quads.append((dy, dx, sy, sx, uv))
                for side in _EDGE_SIDES:
                    self._edges.append(g.sloped_tile_edge_indices(self.tile_px, side, *key[:4]))
                self.corners.append(key)
                self.ids[key] = sid
                self._packed = None
            out.append(sid)
        return out

    def packed(self) -> tuple[tuple, tuple, tuple[np.ndarray, np.ndarray]]:
        """(quad Ops, edge Ops, blend_samples), rebuilt together whenever a
        shape is registered, so the samples always cover the quads."""
        if self._packed is None:
            quads = _csr([q[:4] for q in self._quads], src=True)
            uv = _i32([q[4] for q in self._quads])
            dsy, dsx = self._diamond_src
            self._packed = (
                (quads["dy"], quads["dx"], quads["sy"], quads["sx"], uv, quads["start"], quads["ext"], None, None),
                _pack_ops(self._edges),
                _blend_samples(self.tile_px, [dsy, quads["sy"]], [dsx, quads["sx"]]),
            )
        return self._packed

    @property
    def blend_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Every (sy, sx) a registered quad reads (the kernel's _op_shade),
        plus the diamond's."""
        return self.packed()[2]


def _pack_ops(ops: list[tuple]) -> tuple:
    """Destination-only ops (solid paints) as a kernel Ops tuple."""
    c = _csr([op[:2] for op in ops], src=False)
    return (c["dy"], c["dx"], None, None, None, c["start"], c["ext"], None, None)


@lru_cache(maxsize=16)
def sloped_shapes(tile_px: int) -> SlopedShapes:
    return SlopedShapes(tile_px)


# ---------------------------------------------------------------------------
# Priming (TASK-031.85)

# prime_walk()'s batch per step, from the TASK-031.85 atom census: each batch <= 2 ms at 128 px.
PRIME_MASK_BATCH = 8
PRIME_SHAPE_BATCH = 12
PRIME_SHADE_BATCH = 32
PRIME_BAND_ROWS = 32
# KeyMap keys read per step while collecting mask arguments.
PRIME_KEY_SCAN = 512


def prime_walk(tile_px: int, elev_step: int, *, sloped: bool = False, blend_keys=None, heights=None):
    """Fills the caches a first native prepare at one level reads, one atom
    per step, by calling the same producers and factories the prepare calls:
    a generator that yields before each atom, so its first step does nothing.
    Output can't change: it only runs a cached builder earlier, and a prepare
    that comes first, or after an eviction, builds as it does today.

    Stepped (and iso Flat): the fixed ops and one rise per step, the
    stepped_tables() assembly, then its factors (kernel_tables()). Sloped:
    the sloped_shapes() registry. Both then the blend masks: blend_keys()
    returns the KeyMap's keys (None with blending off), snapshotted here, and
    each key's get_blend_mask() pairs are filled PRIME_MASK_BATCH a step, for
    keys whose layer mask is absent or already loaded. A layer mask load is
    never run here. Sloped last: heights() returns (corner_rise, fold); the
    level's shapes are discovered PRIME_BAND_ROWS tile rows a step by
    prepare_sloped()'s own encoding, registered PRIME_SHAPE_BATCH a step,
    packed in one step, and shaded PRIME_SHADE_BATCH a step at elev_step.
    crop_source()'s wrap copies are not warmed."""
    yield
    if sloped:
        sloped_shapes(tile_px)
    else:
        _fixed_ops(tile_px)
        _rise_ops(tile_px, elev_step)
        for r in range(2, MAX_RISE + 1):
            yield
            _rise_ops(tile_px, r * elev_step)
        yield
        tables = stepped_tables(tile_px, elev_step)
        yield
        tables.kernel_tables()
    yield
    keys = None if blend_keys is None else blend_keys()
    if keys:
        yield from _prime_masks(list(keys), tile_px)
    if sloped:
        yield from _prime_shapes(tile_px, elev_step, heights)


def _prime_masks(keys: list, tile_px: int):
    """prime_walk()'s blend masks over a snapshot of KeyMap keys."""
    seen: set = set()
    pending: list = []
    for i in range(0, len(keys), PRIME_KEY_SCAN):
        if i:
            yield
        for key in keys[i:i + PRIME_KEY_SCAN]:
            layer, pairs = blend_mask_args(key, tile_px)
            if layer is not None and not asset_source.terrain_mask_loaded(*layer):
                continue
            for args in pairs:
                if args not in seen:
                    seen.add(args)
                    pending.append(args)
        while len(pending) >= PRIME_MASK_BATCH:
            yield
            for args in pending[:PRIME_MASK_BATCH]:
                asset_source.get_blend_mask(*args)
            del pending[:PRIME_MASK_BATCH]
    if pending:
        yield
        for args in pending:
            asset_source.get_blend_mask(*args)


def _prime_shapes(tile_px: int, elev_step: int, heights):
    """prime_walk()'s Sloped discovery, registration, pack and shades."""
    yield
    corner_rise, fold = heights()
    rows = corner_rise.shape[0] - 1
    registered = set(sloped_shapes(tile_px).ids)
    seen: set = set()
    found: list = []
    for y0 in range(0, rows, PRIME_BAND_ROWS):
        if y0:
            yield
        y1 = min(rows, y0 + PRIME_BAND_ROWS)
        band = corner_rise[y0:y1 + 1]
        keys, _inv = _shape_keys(
            band[:-1, :-1].ravel(), band[:-1, 1:].ravel(), band[1:, :-1].ravel(), band[1:, 1:].ravel(),
            None if fold is None else fold[y0:y1].ravel(),
        )
        for key in keys:
            if key not in seen:
                seen.add(key)
                found.append(key)
    new = [key for key in found if key not in registered]
    for i in range(0, len(new), PRIME_SHAPE_BATCH):
        yield
        sloped_shapes(tile_px).shape_ids(new[i:i + PRIME_SHAPE_BATCH])
    yield
    sloped_shapes(tile_px).packed()
    # Every shape, not only new ones: an earlier registration may have shaded at another elev_step.
    for i in range(0, len(found), PRIME_SHADE_BATCH):
        yield
        for key in found[i:i + PRIME_SHADE_BATCH]:
            render_sloped._slope_shade(tile_px, *key[:4], elev_step)


# ---------------------------------------------------------------------------
# Unit pack


class UnitPack:
    """One level's marks, sprite draws and farm tiles as flat arrays the
    kernel walks per candidate, in render_iso/render_sloped._paint_objects_*'s own order: a
    tile's marks (units_by_tile order, minus sprites.skip_ids), then its
    sprites (by_anchor order). Each row ends in its unit's base rise, the
    value the Python path records in its ink plane. Farm override ids feed
    the texture choice in Python; the outline mask and colour go to the kernel.
    A multi-tile sprite's column slices (TASK-221) are ordinary sprite rows:
    a strided view, copied contiguous once per distinct rgba by _slot(), at
    its own anchor tile, so neither the row layout nor the kernel changed.

    Rows live in an append-only arena: index[t] = (start, count). A tile is
    derived lazily, the first time a composite's candidates include it
    (ensure()), so a new pack costs nothing up front. A pooled batch covering
    enough tiles derives all pending() tiles source-wise (derive_pending()).
    A level warm may derive them earlier, in idle-time slices through
    ensure() (render_cache's pack_warm_job). refresh() just marks tiles for
    re-derivation, orphaning their old rows until a compaction.

    A pack describes exactly the (units_by_tile, sprites, version, heights)
    it was built or last refreshed against. matches() is the owning cache's
    staleness check, and an owner that can't prove a pack current drops it
    for a new one, never a refresh. heights is elevations (Stepped:
    a mark sits at its own tile's elevation) or corner_rise (Sloped: at
    unit_rise_px), and each owner refreshes the tiles of every unit whose
    height input an edit moved, the same set its SpriteLayer re-anchors."""

    def __init__(
        self, sloped: bool, w: int, h: int, proj, units_by_tile: dict, sprites, version: int, heights, fold=None
    ):
        self.sloped = sloped
        self.w, self.h = w, h
        self.proj = proj
        self.index = np.zeros(h * w * 2, dtype=np.int32)
        self.rows = np.zeros((1024, ROW_W), dtype=np.int32)
        self.n = 0
        self.garbage = 0
        self.farm_tid = np.full(h * w, -1, dtype=np.int32)
        self.farm_mask = np.zeros(h * w, dtype=np.uint8)
        self.farm_rgb = np.zeros(h * w * 3, dtype=np.uint8)
        self.has_farms = False
        self._slots: dict[int, int] = {}
        self._rgbas: list[tuple[np.ndarray, np.ndarray]] = []
        self.addrs = np.zeros(64, dtype=np.uint64)
        self.dims = np.zeros(128, dtype=np.int32)
        self.ready = np.zeros(h * w, dtype=bool)
        # A superset of the tiles holding anything, so ensure() spends no
        # Python on the rest.
        self.has = np.zeros(h * w, dtype=bool)
        # Whether a worker may be reading index/farm_mask/farm_rgb; see kernel_units().
        self._shared = False
        keys = list(units_by_tile)
        if sprites is not None:
            keys += list(sprites.by_anchor)
            keys += list(sprites.farm_by_tile)
        if keys:
            xy = np.array(keys, dtype=np.int64).reshape(-1, 2)
            on_map = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            self.has[xy[on_map, 1] * w + xy[on_map, 0]] = True
        self._bind(units_by_tile, sprites, version, heights, fold)

    def _bind(self, units_by_tile, sprites, version, heights, fold) -> None:
        self._ubt = units_by_tile
        self._sprites = sprites
        self._version = version
        self._heights = heights
        self._fold = fold

    def matches(self, units_by_tile, sprites, version: int, heights) -> bool:
        return (
            self._ubt is units_by_tile and self._sprites is sprites
            and self._version == version and self._heights is heights
        )

    def refresh(self, tiles, units_by_tile, sprites, version: int, heights, fold=None) -> None:
        """Rebinds to the new sources and marks `tiles` for re-derivation.
        Only valid when the pack matched the pre-edit sources and `tiles`
        covers every tile whose rows the edit could have changed."""
        self._bind(units_by_tile, sprites, version, heights, fold)
        w, h = self.w, self.h
        flat = [y * w + x for x, y in tiles if 0 <= x < w and 0 <= y < h]
        if flat:
            self.ready[flat] = False
            self.has[flat] = True

    def pending(self) -> np.ndarray:
        """Flat indices of the tiles that hold something and aren't derived yet."""
        return np.flatnonzero(self.has & ~self.ready)

    def ensure(self, idx: np.ndarray) -> None:
        """Derives every not-yet-derived tile among flat indices `idx`, which
        must be every tile the coming composite reads."""
        need = idx[~self.ready[idx]]
        if need.size == 0:
            return
        need = np.unique(need)
        self.ready[need] = True
        # A tile never marked in `has` was never written either: nothing to clear.
        need = need[self.has[need]]
        self._write_tiles(zip((need % self.w).tolist(), (need // self.w).tolist(), strict=True))
        if self.garbage > max(self.n - self.garbage, 65536):
            self._compact()

    def derive_pending(self) -> None:
        """Derives dense pending sets source-wise, sparse refreshes by tile lookup."""
        need = self.pending()
        if not need.size:
            return
        sprites = self._sprites
        source_tiles = len(self._ubt)
        if sprites is not None:
            source_tiles += len(sprites.by_anchor) + len(sprites.farm_by_tile)
        # Bound the source scan to four entries per pending tile; sparse edits keep tile-local lookups.
        if 4 * need.size < source_tiles:
            self.ensure(need)
            return
        pending = np.zeros(self.w * self.h, dtype=bool)
        pending[need] = True
        w, h = self.w, self.h
        skip = sprites.skip_ids if sprites is not None else frozenset()
        keys, gathered = [], []
        mark_row = self._mark_row
        for (x, y), units in self._ubt.items():
            if not (0 <= x < w and 0 <= y < h):
                continue
            t = y * w + x
            if pending[t]:
                for unit, color in units:
                    if id(unit) not in skip:
                        keys.append(t * 2)
                        gathered.append(mark_row(unit, color, (x, y)))
        farm_ts, farm_vals = [], []
        if sprites is not None:
            for (x, y), draws in sprites.by_anchor.items():
                if not (0 <= x < w and 0 <= y < h):
                    continue
                t = y * w + x
                if pending[t]:
                    for draw, ax, ay, rise in draws:
                        keys.append(t * 2 + 1)
                        gathered.append((ROW_SPRITE, ay - draw.hotspot_y, ax - draw.hotspot_x, draw.rgba, rise))
            for (x, y), farm in sprites.farm_by_tile.items():
                if 0 <= x < w and 0 <= y < h and pending[y * w + x]:
                    farm_ts.append(y * w + x)
                    farm_vals.append(farm)
        keys = np.asarray(keys, dtype=np.int64)
        counts = np.bincount(keys // 2, minlength=w * h)[need]
        rows = []
        slot_of = self._slots.get
        # Stable tile/phase order also pins first-use slot ids to ensure()'s order.
        for i in np.argsort(keys, kind="stable").tolist():
            row = gathered[i]
            if row[0] == ROW_SPRITE:
                rgba = row[3]
                slot = slot_of(id(rgba))
                if slot is None:
                    slot = self._slot(rgba)
                row = (row[0], row[1], row[2], slot, row[4])
            rows.append(row)
        self._write_rows(need, counts, rows, farm_ts, farm_vals)
        self.ready[need] = True
        if self.garbage > max(self.n - self.garbage, 65536):
            self._compact()

    def _slot(self, rgba: np.ndarray) -> int:
        slot = self._slots.get(id(rgba))
        if slot is None:
            flat = np.ascontiguousarray(rgba, dtype=np.uint8)
            if flat.ndim != 3 or flat.shape[2] != 4:
                raise ValueError(f"sprite rgba must be (h, w, 4), got {flat.shape}")
            slot = len(self._rgbas)
            # The original is held too, so its id() can't be reused while mapped.
            self._rgbas.append((rgba, flat))
            self._slots[id(rgba)] = slot
            if slot >= self.addrs.size:
                self.addrs = np.concatenate([self.addrs, np.zeros(self.addrs.size, dtype=np.uint64)])
                self.dims = np.concatenate([self.dims, np.zeros(self.dims.size, dtype=np.int32)])
            self.addrs[slot] = flat.ctypes.data
            self.dims[2 * slot] = flat.shape[0]
            self.dims[2 * slot + 1] = flat.shape[1]
        return slot

    def _mark_row(self, unit, color, key) -> tuple:
        packed = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
        if self.sloped:
            base, rise = render_sloped._unit_mark_sloped(unit, key[0], key[1], self.proj, self._heights, self._fold)
        else:
            base = render_units._unit_mark_base_iso(unit, key[0], key[1], self.proj, self._heights)
            rise = render_units._unit_mark_rise_iso(unit, self.proj, self._heights)
        if base is None:
            return ROW_CONFORM, 0, 0, packed, rise
        return ROW_MARK, base[0], base[1], packed, rise

    def _tile_rows(self, key, skip, ubt_get, anchor_get, rows: list) -> None:
        """Appends tile `key`'s rows to `rows`: marks, then sprites."""
        for unit, color in ubt_get(key, ()):
            if id(unit) not in skip:
                rows.append(self._mark_row(unit, color, key))
        slot_of = self._slots.get
        for draw, ax, ay, rise in anchor_get(key, ()):
            rgba = draw.rgba
            slot = slot_of(id(rgba))
            if slot is None:
                slot = self._slot(rgba)
            rows.append((ROW_SPRITE, ay - draw.hotspot_y, ax - draw.hotspot_x, slot, rise))

    def _write_tiles(self, tiles) -> None:
        sprites = self._sprites
        skip = sprites.skip_ids if sprites is not None else frozenset()
        by_anchor = sprites.by_anchor if sprites is not None else {}
        farms = sprites.farm_by_tile if sprites is not None else {}
        w, h = self.w, self.h
        # Rows are gathered in Python and written in one vectorized step; a
        # numpy write per tile was most of a full build's cost. `tiles` must
        # not repeat a tile.
        ts: list[int] = []
        counts: list[int] = []
        rows: list[tuple[int, int, int, int, int]] = []
        farm_ts: list[int] = []
        farm_vals: list[tuple] = []
        ubt_get, anchor_get, farm_get, tile_rows = self._ubt.get, by_anchor.get, farms.get, self._tile_rows
        for key in tiles:
            tx, ty = key
            if not (0 <= tx < w and 0 <= ty < h):
                continue
            t = ty * w + tx
            n0 = len(rows)
            tile_rows(key, skip, ubt_get, anchor_get, rows)
            ts.append(t)
            counts.append(len(rows) - n0)
            farm = farm_get(key)
            if farm is not None:
                farm_ts.append(t)
                farm_vals.append(farm)
        self._write_rows(ts, counts, rows, farm_ts, farm_vals)

    def _write_rows(self, ts, counts, rows, farm_ts, farm_vals) -> None:
        """Appends gathered rows and replaces each tile's index and farm data."""
        if len(ts) == 0:
            return
        self._unshare()
        t_arr = np.asarray(ts, dtype=np.int64)
        c_arr = np.asarray(counts, dtype=np.int64)
        self.garbage += int(self.index[2 * t_arr + 1].sum())
        starts = np.zeros(c_arr.size, dtype=np.int64)
        np.cumsum(c_arr[:-1], out=starts[1:])
        need = self.n + len(rows)
        if need > self.rows.shape[0]:
            grown = np.zeros((max(need, 2 * self.rows.shape[0]), ROW_W), dtype=np.int32)
            grown[: self.n] = self.rows[: self.n]
            self.rows = grown
        if rows:
            self.rows[self.n : need] = np.array(rows, dtype=np.int32)
        self.index[2 * t_arr] = np.where(c_arr > 0, starts + self.n, 0)
        self.index[2 * t_arr + 1] = c_arr
        self.n = need
        self.farm_tid[t_arr] = -1
        self.farm_mask[t_arr] = 0
        # The kernel never reads it behind a cleared mask; zeroed so a refreshed pack equals a fresh one.
        self.farm_rgb.reshape(-1, 3)[t_arr] = 0
        for t, (terrain_id, color, mask) in zip(farm_ts, farm_vals, strict=True):
            self.farm_tid[t] = terrain_id
            self.farm_mask[t] = mask
            self.farm_rgb[3 * t : 3 * t + 3] = color
            self.has_farms = True

    def _compact(self) -> None:
        self._unshare()
        starts = self.index[0::2].astype(np.int64)
        counts = self.index[1::2].astype(np.int64)
        live = np.nonzero(counts)[0]
        total = int(counts[live].sum())
        new_starts = np.zeros(live.size, dtype=np.int64)
        np.cumsum(counts[live][:-1], out=new_starts[1:])
        src = np.repeat(starts[live] - new_starts, counts[live]) + np.arange(total)
        self.rows = np.ascontiguousarray(self.rows[src]) if total else np.zeros((1024, ROW_W), dtype=np.int32)
        self.index[2 * live] = new_starts
        self.n = total
        self.garbage = 0

    @property
    def two_pass(self) -> bool:
        """Whether a composite over this pack runs terrain then objects (sprites
        on), as render_iso/render_sloped._paint_candidates_* decides it: from the sprite layer."""
        return self._sprites is not None

    def _unshare(self) -> None:
        """Copy-on-write for the arrays rewritten in place. rows needs none:
        appends land past every shared reader's n, and growth or compaction
        builds a new array while the reader's tuple keeps the old one alive."""
        if self._shared:
            self.index = self.index.copy()
            self.farm_mask = self.farm_mask.copy()
            self.farm_rgb = self.farm_rgb.copy()
            self._shared = False

    def kernel_units(self, share: bool = False) -> tuple:
        """share=True hands the arrays to a worker thread (margin_warm), so the
        next in-place write copies them first instead of racing the reader.
        A torn index read there is an out-of-bounds rows read, not a stale pixel."""
        if share:
            self._shared = True
        return (self.index, self.rows[: max(self.n, 1)].reshape(-1), self.addrs, self.dims, self.farm_mask, self.farm_rgb)


# ---------------------------------------------------------------------------
# Per-call inputs


_TERRAIN_ID = attrgetter("terrain_id")


def _terrain_ids(scenario, idx: np.ndarray) -> np.ndarray:
    # list.__getitem__ skips UuidList's pure-Python override, halving the read.
    terrain = scenario.map_manager.terrain
    tiles = map(list.__getitem__, repeat(terrain, idx.size), idx.tolist())
    return np.fromiter(map(_TERRAIN_ID, tiles), dtype=np.int64, count=idx.size)


# Upper bound for _dense_unique over terrain ids (farm override ids included).
TERRAIN_ID_BOUND = 4096


def _dense_unique(values: np.ndarray, bound: int) -> tuple[np.ndarray, np.ndarray]:
    """np.unique(values, return_inverse=True) with a flat inverse, via a
    presence mask and cumsum remap for values in [0, bound)."""
    flat = values.reshape(-1)
    if flat.size == 0 or int(flat.min()) < 0 or int(flat.max()) >= bound:
        uniq, inverse = np.unique(flat, return_inverse=True)
        return uniq, inverse.reshape(-1)
    present = np.zeros(bound, dtype=bool)
    present[flat] = True
    remap = np.cumsum(present, dtype=np.intp) - 1
    return np.flatnonzero(present).astype(values.dtype, copy=False), remap[flat]


def _textures(
    tids: np.ndarray, tile_px: int, textures: bool, extra: list[int] | None = None
) -> tuple[np.ndarray, tuple, np.ndarray] | None:
    """(per-candidate slot, (texture list, flat colours, sizes), the sorted
    terrain id of each slot), or None for a texture the kernel can't crop the
    way the numpy path does. extra: more terrain ids to give slots (blend
    layers and neighbours) beyond the candidates' own."""
    if extra:
        tids = np.concatenate([tids, np.asarray(extra, dtype=tids.dtype)])
    uniq, slots = _dense_unique(tids, TERRAIN_ID_BOUND)
    if extra:
        slots = slots[: slots.size - len(extra)]
    arrays: list = []
    flat = np.zeros((max(uniq.size, 1), 3), dtype=np.uint8)
    sizes = np.zeros(max(uniq.size, 1), dtype=np.int64)
    size = asset_source.terrain_texture_size(tile_px)
    for i, tid in enumerate(uniq.tolist()):
        tex = asset_source.terrain_texture_at(tid, size) if textures else None
        if tex is None:
            flat[i] = color_for_terrain_id(tid)
            arrays.append(None)
            continue
        render_paint._crop_offset(0, 0, tex.shape[0], tile_px)  # the numpy path's own bound assert
        if tex.ndim != 3 or tex.shape[0] != tex.shape[1] or tex.shape[2] != 3:
            return None
        # The kernel crops at (x * tile_px) % size, size the period, stride the padded width.
        arrays.append(np.ascontiguousarray(asset_source.crop_source(tex, tile_px), dtype=np.uint8))
        sizes[i] = tex.shape[0]
    return slots.astype(np.int32).reshape(-1), (arrays, flat, sizes), uniq


def blend_mask_args(key, tile_px: int) -> tuple[tuple[str, int] | None, list[tuple[int, int, int]]]:
    """What _blend_tables() fetches for one KeyMap key at tile_px: the layer
    mask's terrain_mask_at() arguments (None without a layer) and each
    neighbour entry's get_blend_mask() arguments, in entry order. One
    helper, so prime_walk() warms exactly the keys a prepare reads."""
    _own, layer_tid, entries = key
    layer = None
    if layer_tid >= 0:
        layer = (terrain_blend.layer_mask_name(layer_tid), asset_source.terrain_texture_size(tile_px))
    return layer, [(mode, tile, tile_px) for _n_tid, mode, tile in entries]


# Blend op kinds, mirrored by DEF constants in _composite_native.pyx.
BLEND_LAYER, BLEND_NEIGHBOUR = 0, 1
# _blend_tables()' answer for a blend the kernel can't compose byte-identically.
UNCOMPOSABLE = object()


def _blend_tables(blend: terrain_blend.RectKeys, base_tids: np.ndarray, overridden: np.ndarray | None,
                  tile_px: int, tex_tables: tuple, uniq: np.ndarray):
    """The kernel's blend input for render_paint's per-candidate blend keys: (per
    candidate spec id or -1, spec start offsets, (kind, texture slot, mask
    id) op triples, masks). A candidate composes only where render_paint._top_block
    would: a key, no farm override, and the key's own tid equal to the tid
    its plain crop reads. The ops are built once per distinct key among
    those candidates, and the spec ids here number that subset (not the
    KeyMap's own). The kernel skips an op whose texture slot is a
    flat colour or whose mask id is -1, as terrain_blend._compose skips a
    missing texture or mask. None when no candidate blends; UNCOMPOSABLE
    (take the numpy path) when a texture or mask is not the one size every
    crop needs."""
    sizes = tex_tables[2]
    size = {int(s) for s in sizes if s}
    ids, table = blend
    valid = ids >= 0
    # Never own[-1]: a keyless candidate reads spec 0's tid and stays invalid.
    valid &= table.own[np.where(valid, ids, 0)] == base_tids
    if overridden is not None:
        valid &= ~overridden
    specs, inverse = np.unique(ids[valid], return_inverse=True)
    if not specs.size:
        return None
    cand = np.full(base_tids.size, -1, dtype=np.int32)
    cand[valid] = inverse.reshape(-1)
    ops: list[tuple[int, int, int]] = []
    starts = [0]
    masks: list[np.ndarray] = []
    mask_of: dict = {}
    slot_of = {int(t): i for i, t in enumerate(uniq.tolist())}

    def mask_id(name, arr) -> int:
        if arr is None:
            return -1
        mid = mask_of.get(name)
        if mid is None:
            mid = mask_of[name] = len(masks)
            masks.append(np.ascontiguousarray(arr, dtype=np.uint8))
        return mid

    keys = table.keys
    for sid in specs.tolist():
        key = keys[sid]
        layer, pairs = blend_mask_args(key, tile_px)
        if layer is not None:
            mask = asset_source.terrain_mask_at(*layer)
            mask = None if mask is None else asset_source.crop_source(mask, tile_px)
            ops.append((BLEND_LAYER, slot_of[key[1]], mask_id(("layer", layer[0]), mask)))
        for (n_tid, _mode, _tile), args in zip(key[2], pairs, strict=True):
            ops.append((BLEND_NEIGHBOUR, slot_of[n_tid], mask_id(args[:2], asset_source.get_blend_mask(*args))))
        starts.append(len(ops))
    if len(size) > 1:
        return UNCOMPOSABLE
    if size:
        (tsize,) = size
        padded = tsize + asset_source.crop_pad(tsize, tile_px)
        for kind, _slot, mid in ops:
            want = (padded, padded) if kind == BLEND_LAYER else (tile_px, tile_px)
            if mid >= 0 and masks[mid].shape != want:
                return UNCOMPOSABLE
    return (
        cand, np.array(starts, dtype=np.int64),
        np.array(ops, dtype=np.int32).reshape(-1) if ops else np.zeros(0, dtype=np.int32), masks,
    )


def _blend_extra_tids(blend: terrain_blend.RectKeys | None) -> list[int]:
    """Every layer and neighbour terrain id the candidates' keys name
    (farm-overridden candidates included), for _textures()."""
    if blend is None:
        return []
    ids, table = blend
    extra: set[int] = set()
    for sid in np.unique(ids[ids >= 0]).tolist():
        extra.update(table.extra[sid])
    return sorted(extra)


def _edge_states(
    cx: np.ndarray, cy: np.ndarray, w: int, h: int, draw_minors: bool, elevations, major_every: int,
) -> np.ndarray:
    """grid_overlay.owned_edges per candidate as one base-3 key over
    (x_low, y_low, x_high, y_high): 0 absent, 1 minor, 2 major, with the LOD
    minor filter already applied."""

    def state(n, present=True):
        s = np.where(n % major_every == 0, 2, 1)
        s = np.where(present, s, 0)
        return s if draw_minors else np.where(s == 1, 0, s)

    if elevations is None:
        xh_present = cx + 1 == w
        yh_present = cy + 1 == h
    else:
        here = elevations[cy, cx]
        xh_present = (cx + 1 == w) | (elevations[cy, np.minimum(cx + 1, w - 1)] != here)
        yh_present = (cy + 1 == h) | (elevations[np.minimum(cy + 1, h - 1), cx] != here)
    return ((state(cx) * 3 + state(cy)) * 3 + state(cx + 1, xh_present)) * 3 + state(cy + 1, yh_present)


_EDGE_NAMES = ("x_low", "y_low", "x_high", "y_high")


def _edges_for_key(key: int) -> tuple:
    states = []
    for _ in range(4):
        states.append(key % 3)
        key //= 3
    states.reverse()
    return tuple((name, s == 2) for name, s in zip(_EDGE_NAMES, states, strict=True) if s)


def _grid_stamps(tile_px: int, grid, keys: np.ndarray, split=None) -> tuple[np.ndarray, tuple | None]:
    """(per-candidate stamp id or -1, stamp tables). keys are per-candidate
    ints: an _edge_states key for Stepped, or a pair id that `split` maps to
    (edge key, normalized Sloped corners). Stamps come from
    render_paint._grid_stamp_iso's lru on every call."""
    # A Stepped key is 4 base-3 digits; Sloped pair ids have no tight bound.
    uniq, inv = _dense_unique(keys, 81) if split is None else np.unique(keys, return_inverse=True)
    parts, sid_of = [], np.full(uniq.size, -1, dtype=np.int32)
    for i, key in enumerate(uniq.tolist()):
        edge_key, corners = (key, None) if split is None else split(key)
        edges = _edges_for_key(edge_key)
        if not edges:
            continue
        dst_y, dst_x, target, alpha, extent = render_paint._grid_stamp_iso(tile_px, edges, grid, corners)
        if extent is None:
            continue
        sid_of[i] = len(parts)
        parts.append((dst_y, dst_x, target, alpha))
    cand = sid_of[inv.reshape(-1)]
    if not parts:
        return np.full(keys.size, -1, dtype=np.int32), None
    c = _csr([p[:2] for p in parts], src=False)
    tgt = np.ascontiguousarray(np.concatenate([p[2] for p in parts]), dtype=np.uint16).reshape(-1)
    alpha = np.ascontiguousarray(np.concatenate([p[3] for p in parts]), dtype=np.uint16).reshape(-1)
    return np.ascontiguousarray(cand, dtype=np.int32), (c["dy"], c["dx"], tgt, alpha, c["start"], c["ext"])


def _candidate_arrays(candidates: np.ndarray, w: int):
    cx = np.ascontiguousarray(candidates[:, 0], dtype=np.int64)
    cy = np.ascontiguousarray(candidates[:, 1], dtype=np.int64)
    return cx, cy, cy * w + cx


def _units_of(pack) -> UnitPack | None:
    return None if pack is None or pack is NO_UNITS else pack


def _tex_ids(scenario, idx, pack: UnitPack | None, terrain_ids: np.ndarray | None = None):
    """(texture tid per candidate, the tid before any farm override, which
    candidates a farm overrides or None)."""
    base = _terrain_ids(scenario, idx) if terrain_ids is None else terrain_ids[idx]
    if pack is not None:
        pack.ensure(idx)
    if pack is not None and pack.has_farms:
        farm = pack.farm_tid[idx]
        overridden = farm >= 0
        return np.where(overridden, farm, base), base, overridden
    return base, base, None


def _terrain_inputs(scenario, idx, pack, terrain_ids, tile_px: int, textures: bool,
                    blend: terrain_blend.RectKeys | None):
    """(slots, texture tables, blend tables or None), or None for a texture
    the kernel can't crop or a blend it can't compose (the caller's numpy
    path)."""
    tids, base, overridden = _tex_ids(scenario, idx, pack, terrain_ids)
    tex = _textures(tids, tile_px, textures, _blend_extra_tids(blend))
    if tex is None:
        return None
    slots, tex_tables, uniq = tex
    tables = None
    if blend is not None and textures:
        tables = _blend_tables(blend, base, overridden, tile_px, tex_tables, uniq)
        if tables is UNCOMPOSABLE:
            return None
    return slots, tex_tables, tables


def composite_iso(native, scratch, scenario, x0, y0, candidates, elevations, proj, tile_px,
                  pack: UnitPack | None, textures: bool, grid, terrain_ids: np.ndarray | None = None,
                  blend: terrain_blend.RectKeys | None = None) -> bool:
    """render_iso.composite_rect_iso's loop, natively. False (nothing painted)
    means the caller must take the per-tile path."""
    args = prepare_iso(
        scenario, x0, y0, candidates, elevations, proj, tile_px, pack, textures, grid, terrain_ids=terrain_ids,
        blend=blend,
    )
    return args is not None and native.composite_iso(scratch, *args)


def prepare_iso(scenario, x0, y0, candidates, elevations, proj, tile_px, pack: UnitPack | None,
                textures: bool, grid, share: bool = False, terrain_ids: np.ndarray | None = None,
                blend: terrain_blend.RectKeys | None = None, snapshot: np.ndarray | None = None) -> tuple | None:
    """native.composite_iso's arguments after the scratch, or None for the
    per-tile path. share=True is for a call on another thread: elevations is
    copied and the pack is marked shared (UnitPack.kernel_units). snapshot: a
    copy of elevations the caller already owns (a pooled paint batch takes one
    for all its jobs), handed to the kernel as is instead. terrain_ids
    is the owning cache's per-tile mirror, None to read the tiles live.
    blend: render_paint's terrain_blend.RectKeys for the candidates (spec ids into
    the caller's KeyMap, read on this thread before any worker runs), or None;
    the kernel composes each blended top block itself (_blend_tables(), plus
    the geometry's blend_samples: only the texels a paint reads). The last
    argument is the pack's two_pass: terrain then objects, when sprites are on."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    pack = _units_of(pack)
    cx, cy, idx = _candidate_arrays(candidates, w)
    terrain = _terrain_inputs(scenario, idx, pack, terrain_ids, tile_px, textures, blend)
    if terrain is None:
        return None
    slots, tex_tables, blend_tables = terrain
    stamps = None
    cand_grid = np.full(cx.size, -1, dtype=np.int32)
    if grid.paints:
        draw_grid, draw_minors = render_paint._grid_lod_edges_filter(tile_px / 2, grid.major_every)
        if draw_grid:
            keys = _edge_states(cx, cy, w, h, draw_minors, elevations, grid.major_every)
            cand_grid, stamps = _grid_stamps(tile_px, grid, keys)
    tables = stepped_tables(tile_px, proj.elev_step)
    if blend_tables is not None:
        blend_tables = (*blend_tables, *tables.blend_samples)
    geom = (proj.origin_x, proj.origin_y, proj.half_w, proj.half_h, proj.elev_step, tile_px, MAX_RISE)
    # The bbox functions write elevations in place; a worker reads a copy.
    if snapshot is not None:
        elev = snapshot
    else:
        elev = elevations.copy() if share else elevations
    return (
        x0, y0, cx, cy, slots, cand_grid, elev, geom, tables.kernel_tables(), tex_tables,
        stamps, pack.kernel_units(share) if pack is not None else None, blend_tables,
        pack is not None and pack.two_pass,
    )


def composite_sloped(native, scratch, scenario, x0, y0, candidates, corner_rise, proj, tile_px,
                     pack: UnitPack | None, textures: bool, grid, terrain_ids: np.ndarray | None = None,
                     blend: terrain_blend.RectKeys | None = None, fold: np.ndarray | None = None) -> bool:
    """render_sloped.composite_rect_sloped's loop, natively; same False contract."""
    args = prepare_sloped(
        scenario, x0, y0, candidates, corner_rise, proj, tile_px, pack, textures, grid, terrain_ids=terrain_ids,
        blend=blend, fold=fold,
    )
    if args is None:
        return False
    native.composite_sloped(scratch, *args)
    return True


def _shape_keys(nw, ne, sw, se, fold, inverse: bool = False) -> tuple[list, np.ndarray | None]:
    """The distinct normalized Sloped shape keys over 1-D corner arrays (and
    the tiles' fold bits, or None for the fixed NE-SW fold), in scalar order,
    plus each tile's index into them with `inverse`. prepare_sloped()'s
    encoding, shared with prime_walk()'s discovery."""
    dmin = np.minimum(np.minimum(nw, ne), np.minimum(sw, se))
    norm = [nw - dmin, ne - dmin, sw - dmin, se - dmin]
    # The fold bit on non-planar shapes only, so a planar key stays the NE-SW one.
    nwse = np.zeros(nw.size, dtype=np.int64) if fold is None else (fold & (nw + se != ne + sw)).astype(np.int64)
    # One scalar per shape (base past the largest normalized corner), far
    # cheaper to unique than rows.
    base = int(max(int(c.max()) for c in norm)) + 1
    scalar = (((norm[0] * base + norm[1]) * base + norm[2]) * base + norm[3]) * 2 + nwse
    if inverse:
        _uniq, first, inv = np.unique(scalar, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
    else:
        _uniq, first = np.unique(scalar, return_index=True)
        inv = None
    return [(*(int(c[i]) for c in norm), bool(nwse[i])) for i in first.tolist()], inv


def prepare_sloped(scenario, x0, y0, candidates, corner_rise, proj, tile_px, pack: UnitPack | None,
                   textures: bool, grid, share: bool = False, terrain_ids: np.ndarray | None = None,
                   blend: terrain_blend.RectKeys | None = None, fold: np.ndarray | None = None) -> tuple | None:
    """prepare_iso() for native.composite_sloped. corner_rise needs no copy:
    SlopedChunkCache replaces it on every refresh, never writes it in place,
    and its fold (tile_folds(), None for the fixed NE-SW fold) with it."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    pack = _units_of(pack)
    cx, cy, idx = _candidate_arrays(candidates, w)
    terrain = _terrain_inputs(scenario, idx, pack, terrain_ids, tile_px, textures, blend)
    if terrain is None:
        return None
    slots, tex_tables, blend_tables = terrain

    nw, ne = corner_rise[cy, cx], corner_rise[cy, cx + 1]
    sw, se = corner_rise[cy + 1, cx], corner_rise[cy + 1, cx + 1]
    shape_keys, inv = _shape_keys(nw, ne, sw, se, None if fold is None else fold[cy, cx], inverse=True)
    registry = sloped_shapes(tile_px)
    shape_of = np.array(registry.shape_ids(shape_keys), dtype=np.int32)
    cand_shape = np.ascontiguousarray(shape_of[inv], dtype=np.int32)

    shades = [render_sloped._slope_shade(tile_px, *key[:4], proj.elev_step) for key in shape_keys]
    shade_start = np.zeros(len(shades), dtype=np.int64)
    np.cumsum([s.size for s in shades[:-1]], out=shade_start[1:])
    shade = np.ascontiguousarray(np.concatenate(shades), dtype=np.float32)
    cand_shade = np.ascontiguousarray(shade_start[inv], dtype=np.int64)

    stamps = None
    cand_grid = np.full(cx.size, -1, dtype=np.int32)
    if grid.paints:
        draw_grid, draw_minors = render_paint._grid_lod_edges_filter(tile_px / 2, grid.major_every)
        if draw_grid:
            edge_keys = _edge_states(cx, cy, w, h, draw_minors, None, grid.major_every)
            pair = edge_keys * len(shape_keys) + inv
            cand_grid, stamps = _grid_stamps(
                tile_px, grid, pair, split=lambda p: (p // len(shape_keys), shape_keys[p % len(shape_keys)][:4]),
            )
    quads, edges, samples = registry.packed()
    if blend_tables is not None:
        blend_tables = (*blend_tables, *samples)
    geom = (proj.origin_x, proj.origin_y, proj.half_w, proj.half_h, tile_px)
    return (
        x0, y0, cx, cy, slots, cand_grid, cand_shape, cand_shade, corner_rise, geom,
        quads, edges, registry.diamond, shade, tex_tables, stamps,
        pack.kernel_units(share) if pack is not None else None, blend_tables,
        pack is not None and pack.two_pass,
    )
