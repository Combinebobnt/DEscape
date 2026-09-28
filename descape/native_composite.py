"""Batch F N2: composite_rect_iso/_sloped as one native call per rect.

Python keeps everything with Python-rounding or cache semantics and hands the
kernel flat arrays: the ordered candidates (tiles_in_screen_rect plus the
bystander merge), each candidate's texture slot and grid stamp, the index
tables, and a per-level UnitPack of marks, sprites and farms. The numpy path
in render.py stays the byte-identity oracle (tests/test_native_composite.py).

What is read live on every call, never cached here, so nothing can go stale
under an edit: terrain ids (~30ns a tile), elevations and corner_rise, the
textures (from asset_source's own lru, held only for the call), and every
factor, LUT, shade and grid-stamp table (from render's lru caches, which tests
rebind and cache_clear()). Cached across calls: pure index geometry keyed by
(tile_px, elev_step) or by a normalized Sloped shape, and the UnitPack, which
its owning cache refreshes through the same splice funnels that update its
SpriteLayer (see UnitPack).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import repeat
from operator import attrgetter

import numpy as np

from descape import asset_source, iso_geometry, render
from descape.edge_ticks import MAJORS_PER_MINOR

# Stepped op-table layout, mirrored by DEF constants in _composite_native.pyx.
N_FIXED = 8
PER_RISE = 9
MAX_RISE = iso_geometry.MAX_ELEVATION - iso_geometry.MIN_ELEVATION

ROW_MARK, ROW_CONFORM, ROW_SPRITE = 0, 1, 2

# What a with_units=False cache passes as composite_rect_*'s unit_pack: take
# the whole-rect path, with nothing to draw after the terrain.
NO_UNITS = object()
_EDGE_SIDES = tuple(side for _bit, side in render._FARM_EDGE_BITS)
_EMPTY_EXT = (1, 0, 0, 0)


def _extent(dst_y: np.ndarray, dst_x: np.ndarray) -> tuple[int, int, int, int]:
    if dst_y.size == 0:
        return _EMPTY_EXT
    return int(dst_y.min()), int(dst_y.max()), int(dst_x.min()), int(dst_x.max())


def _i32(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=np.int32)
    wide = np.concatenate(parts)
    out = wide.astype(np.int32)
    if not np.array_equal(out, wide):
        raise OverflowError("an index table value does not fit in int32")
    return out


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
        "ext": np.array([_extent(op[0], op[1]) for op in ops], dtype=np.int32).reshape(-1),
    }
    if src:
        cols["sy"] = _i32([op[2] for op in ops])
        cols["sx"] = _i32([op[3] for op in ops])
    return cols


class SteppedTables:
    """Every Stepped index producer's output for one (tile_px, elev_step),
    rises 1..MAX_RISE, in the kernel's op order. Geometry only: the darken
    factors are fetched per call (stepped_factors)."""

    def __init__(self, tile_px: int, elev_step: int):
        g = iso_geometry
        ops: list[tuple] = [g.diamond_indices(tile_px)]
        # factor_keys[i] is the render._seam_factors/_shadow_factors key for
        # a darken op, None for a scatter or solid one.
        keys: list[tuple | None] = [None]
        for side in ("up_left", "up_right"):
            ops.append(g.seam_edge_indices(tile_px, side))
            keys.append(("seam", side))
        ops.append(g.seam_apex_indices(tile_px))
        keys.append(("seam", "apex"))
        for side in _EDGE_SIDES:
            ops.append(g.tile_edge_indices(tile_px, side))
            keys.append(None)
        assert len(ops) == N_FIXED
        for r in range(1, MAX_RISE + 1):
            rise = r * elev_step
            ops.append(g.skirt_quad_indices(tile_px, rise, "left"))
            keys.append(None)
            ops.append(g.skirt_quad_indices(tile_px, rise, "right"))
            keys.append(None)
            for side in ("up_left", "up_right"):
                ops.append(g.shadow_quad_indices(tile_px, rise, side)[:2])
                keys.append(("shadow", rise, side))
            for sides in ("both", "up_left", "up_right"):
                ops.append(g.shadow_apex_indices(tile_px, rise, sides)[:2])
                keys.append(("shadow", rise, "apex_" + sides))
            for side in ("up_left", "up_right"):
                ops.append(g.shadow_tip_indices(tile_px, rise, side)[:2])
                keys.append(("shadow", rise, "tip_" + side))
        assert len(ops) == N_FIXED + MAX_RISE * PER_RISE
        # One src entry per dst entry in every op, zeros where an op samples
        # nothing, so the four columns share the start offsets.
        cols = _csr([op if len(op) == 4 else (*op, np.zeros_like(op[0]), np.zeros_like(op[0])) for op in ops], src=True)
        for name in ("sy", "sx"):
            assert cols[name].size == 0 or (cols[name].min() >= 0 and cols[name].max() < tile_px)
        self.tile_px = tile_px
        self.cols = cols
        self.factor_keys = keys
        self.op_len = np.diff(cols["start"])
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
        """The darken factors, fetched from render's lru caches on every call
        and re-concatenated only when one of them is a different array than
        last time (a test rebinding SEAM_SHADE then cache_clear()ing)."""
        src = []
        for key in self.factor_keys:
            if key is None:
                continue
            if key[0] == "seam":
                src.append(render._seam_factors(self.tile_px, key[1]))
            else:
                src.append(render._shadow_factors(self.tile_px, key[1], key[2]))
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
        luts = np.ascontiguousarray(np.stack([render._skirt_lut("left"), render._skirt_lut("right")]), dtype=np.uint8)
        return (c["dy"], c["dx"], c["sy"], c["sx"], None, c["start"], c["ext"], self.fstart, self.factors(), luts)


@lru_cache(maxsize=16)
def stepped_tables(tile_px: int, elev_step: int) -> SteppedTables:
    return SteppedTables(tile_px, elev_step)


class SlopedShapes:
    """Sloped quad and farm-edge geometry per normalized corner shape, for one
    tile_px. Shapes accumulate as candidates reveal them; both producers are
    normalization-invariant (they subtract the corner minimum themselves), so
    a shape's arrays are the same for every tile that has it."""

    def __init__(self, tile_px: int):
        self.tile_px = tile_px
        self.ids: dict[tuple[int, int, int, int], int] = {}
        self.corners: list[tuple[int, int, int, int]] = []
        self._quads: list[tuple] = []
        self._edges: list[tuple] = []
        self._packed: tuple | None = None
        self.diamond = _pack_ops([iso_geometry.diamond_indices(tile_px)[:2]])

    def shape_ids(self, keys: list[tuple[int, int, int, int]]) -> list[int]:
        out = []
        for key in keys:
            sid = self.ids.get(key)
            if sid is None:
                sid = len(self.corners)
                g = iso_geometry
                dy, dx, sy, sx, uv = g.sloped_quad_indices(self.tile_px, *key)
                assert sy.size == 0 or (sy.min() >= 0 and sy.max() < self.tile_px)
                assert sx.size == 0 or (sx.min() >= 0 and sx.max() < self.tile_px)
                self._quads.append((dy, dx, sy, sx, uv))
                for side in _EDGE_SIDES:
                    self._edges.append(g.sloped_tile_edge_indices(self.tile_px, side, *key))
                self.corners.append(key)
                self.ids[key] = sid
                self._packed = None
            out.append(sid)
        return out

    def packed(self) -> tuple[tuple, tuple]:
        if self._packed is None:
            quads = _csr([q[:4] for q in self._quads], src=True)
            uv = _i32([q[4] for q in self._quads])
            self._packed = (
                (quads["dy"], quads["dx"], quads["sy"], quads["sx"], uv, quads["start"], quads["ext"], None, None),
                _pack_ops(self._edges),
            )
        return self._packed


def _pack_ops(ops: list[tuple]) -> tuple:
    """Destination-only ops (solid paints) as a kernel Ops tuple."""
    c = _csr([op[:2] for op in ops], src=False)
    return (c["dy"], c["dx"], None, None, None, c["start"], c["ext"], None, None)


@lru_cache(maxsize=16)
def sloped_shapes(tile_px: int) -> SlopedShapes:
    return SlopedShapes(tile_px)


# ---------------------------------------------------------------------------
# Unit pack


class UnitPack:
    """One level's marks, sprite draws and farm tiles as flat arrays the
    kernel walks per candidate, in _paint_tile_and_units_*'s own order: a
    tile's marks (units_by_tile order, minus sprites.skip_ids), then its
    sprites (by_anchor order). Farm override ids feed the texture choice in
    Python; the outline mask and colour go to the kernel.

    Rows live in an append-only arena: index[t] = (start, count). A tile is
    derived lazily, the first time a composite's candidates include it
    (ensure()), so a new pack costs nothing up front and a whole-map build
    (20-170ms on the corpus) never lands in one frame. A level warm may
    derive them earlier, in idle-time slices of pending() through the same
    ensure() (render_cache's pack_warm_job). refresh() just marks tiles for
    re-derivation, orphaning their old rows until a compaction.

    A pack describes exactly the (units_by_tile, sprites, version, heights)
    it was built or last refreshed against. matches() is the owning cache's
    staleness check, and an owner that can't prove a pack current drops it
    for a new one, never a refresh. heights is elevations (Stepped:
    a mark sits at its own tile's elevation) or corner_rise (Sloped: at
    unit_rise_px), and each owner refreshes the tiles of every unit whose
    height input an edit moved, the same set its SpriteLayer re-anchors."""

    def __init__(self, sloped: bool, w: int, h: int, proj, units_by_tile: dict, sprites, version: int, heights):
        self.sloped = sloped
        self.w, self.h = w, h
        self.proj = proj
        self.index = np.zeros(h * w * 2, dtype=np.int32)
        self.rows = np.zeros((1024, 4), dtype=np.int32)
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
        self._bind(units_by_tile, sprites, version, heights)

    def _bind(self, units_by_tile, sprites, version, heights) -> None:
        self._ubt = units_by_tile
        self._sprites = sprites
        self._version = version
        self._heights = heights

    def matches(self, units_by_tile, sprites, version: int, heights) -> bool:
        return (
            self._ubt is units_by_tile and self._sprites is sprites
            and self._version == version and self._heights is heights
        )

    def refresh(self, tiles, units_by_tile, sprites, version: int, heights) -> None:
        """Rebinds to the new sources and marks `tiles` for re-derivation.
        Only valid when the pack matched the pre-edit sources and `tiles`
        covers every tile whose rows the edit could have changed."""
        self._bind(units_by_tile, sprites, version, heights)
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

    def _tile_rows(self, key, skip, ubt_get, anchor_get, rows: list) -> None:
        """Appends tile `key`'s rows to `rows`: marks, then sprites."""
        for unit, color in ubt_get(key, ()):
            if id(unit) in skip:
                continue
            packed = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
            if self.sloped:
                base = render._unit_mark_sloped(unit, key[0], key[1], self.proj, self._heights)
            else:
                base = render._unit_mark_base_iso(unit, key[0], key[1], self.proj, self._heights)
            if base is None:
                rows.append((ROW_CONFORM, 0, 0, packed))
            else:
                rows.append((ROW_MARK, base[0], base[1], packed))
        slot_of = self._slots.get
        for draw, ax, ay in anchor_get(key, ()):
            rgba = draw.rgba
            slot = slot_of(id(rgba))
            if slot is None:
                slot = self._slot(rgba)
            rows.append((ROW_SPRITE, ay - draw.hotspot_y, ax - draw.hotspot_x, slot))

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
        rows: list[tuple[int, int, int, int]] = []
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
        if not ts:
            return
        self._unshare()
        t_arr = np.array(ts, dtype=np.int64)
        c_arr = np.array(counts, dtype=np.int64)
        self.garbage += int(self.index[2 * t_arr + 1].sum())
        starts = np.zeros(c_arr.size, dtype=np.int64)
        np.cumsum(c_arr[:-1], out=starts[1:])
        need = self.n + len(rows)
        if need > self.rows.shape[0]:
            grown = np.zeros((max(need, 2 * self.rows.shape[0]), 4), dtype=np.int32)
            grown[: self.n] = self.rows[: self.n]
            self.rows = grown
        if rows:
            self.rows[self.n : need] = np.array(rows, dtype=np.int32)
        self.index[2 * t_arr] = np.where(c_arr > 0, starts + self.n, 0)
        self.index[2 * t_arr + 1] = c_arr
        self.n = need
        self.farm_tid[t_arr] = -1
        self.farm_mask[t_arr] = 0
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
        self.rows = np.ascontiguousarray(self.rows[src]) if total else np.zeros((1024, 4), dtype=np.int32)
        self.index[2 * live] = new_starts
        self.n = total
        self.garbage = 0

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


def _textures(tids: np.ndarray, tile_px: int, textures: bool) -> tuple[np.ndarray, tuple] | None:
    """(per-candidate slot, (texture list, flat colours, sizes)), or None for
    a texture the kernel can't crop the way the numpy path does."""
    uniq, slots = np.unique(tids, return_inverse=True)
    arrays: list = []
    flat = np.zeros((max(uniq.size, 1), 3), dtype=np.uint8)
    sizes = np.zeros(max(uniq.size, 1), dtype=np.int64)
    for i, tid in enumerate(uniq.tolist()):
        tex = asset_source.get_terrain_texture_array(tid) if textures else None
        if tex is None:
            flat[i] = render.color_for_terrain_id(tid)
            arrays.append(None)
            continue
        render._crop_offset(0, 0, tex.shape[0], tile_px)  # the numpy path's own divisibility assert
        if tex.ndim != 3 or tex.shape[0] != tex.shape[1] or tex.shape[2] != 3:
            return None
        arrays.append(np.ascontiguousarray(tex, dtype=np.uint8))
        sizes[i] = tex.shape[0]
    return slots.astype(np.int32).reshape(-1), (arrays, flat, sizes)


def _edge_states(cx: np.ndarray, cy: np.ndarray, w: int, h: int, draw_minors: bool, elevations) -> np.ndarray:
    """grid_overlay.owned_edges per candidate as one base-3 key over
    (x_low, y_low, x_high, y_high): 0 absent, 1 minor, 2 major, with the LOD
    minor filter already applied."""

    def state(n, present=True):
        s = np.where(n % MAJORS_PER_MINOR == 0, 2, 1)
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
    render._grid_stamp_iso's lru on every call."""
    uniq, inv = np.unique(keys, return_inverse=True)
    parts, sid_of = [], np.full(uniq.size, -1, dtype=np.int32)
    for i, key in enumerate(uniq.tolist()):
        edge_key, corners = (key, None) if split is None else split(key)
        edges = _edges_for_key(edge_key)
        if not edges:
            continue
        dst_y, dst_x, target, alpha, extent = render._grid_stamp_iso(tile_px, edges, grid, corners)
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


def _tex_ids(scenario, idx, pack: UnitPack | None) -> np.ndarray:
    tids = _terrain_ids(scenario, idx)
    if pack is not None:
        pack.ensure(idx)
    if pack is not None and pack.has_farms:
        farm = pack.farm_tid[idx]
        tids = np.where(farm >= 0, farm, tids)
    return tids


def composite_iso(native, scratch, scenario, x0, y0, candidates, elevations, proj, tile_px,
                  pack: UnitPack | None, textures: bool, grid) -> bool:
    """render.composite_rect_iso's loop, natively. False (nothing painted)
    means the caller must take the per-tile path."""
    args = prepare_iso(scenario, x0, y0, candidates, elevations, proj, tile_px, pack, textures, grid)
    return args is not None and native.composite_iso(scratch, *args)


def prepare_iso(scenario, x0, y0, candidates, elevations, proj, tile_px, pack: UnitPack | None,
                textures: bool, grid, share: bool = False) -> tuple | None:
    """native.composite_iso's arguments after the scratch, or None for the
    per-tile path. share=True is for a call on another thread: elevations is
    copied and the pack is marked shared (UnitPack.kernel_units)."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    pack = _units_of(pack)
    cx, cy, idx = _candidate_arrays(candidates, w)
    tex = _textures(_tex_ids(scenario, idx, pack), tile_px, textures)
    if tex is None:
        return None
    slots, tex_tables = tex
    stamps = None
    cand_grid = np.full(cx.size, -1, dtype=np.int32)
    if grid.paints:
        draw_grid, draw_minors = render._grid_lod_edges_filter(tile_px / 2)
        if draw_grid:
            keys = _edge_states(cx, cy, w, h, draw_minors, elevations)
            cand_grid, stamps = _grid_stamps(tile_px, grid, keys)
    tables = stepped_tables(tile_px, proj.elev_step)
    geom = (proj.origin_x, proj.origin_y, proj.half_w, proj.half_h, proj.elev_step, tile_px, MAX_RISE)
    # The bbox functions write elevations in place; a worker reads a copy.
    elev = elevations.copy() if share else elevations
    return (
        x0, y0, cx, cy, slots, cand_grid, elev, geom, tables.kernel_tables(), tex_tables,
        stamps, pack.kernel_units(share) if pack is not None else None,
    )


def composite_sloped(native, scratch, scenario, x0, y0, candidates, corner_rise, proj, tile_px,
                     pack: UnitPack | None, textures: bool, grid) -> bool:
    """render.composite_rect_sloped's loop, natively; same False contract."""
    args = prepare_sloped(scenario, x0, y0, candidates, corner_rise, proj, tile_px, pack, textures, grid)
    if args is None:
        return False
    native.composite_sloped(scratch, *args)
    return True


def prepare_sloped(scenario, x0, y0, candidates, corner_rise, proj, tile_px, pack: UnitPack | None,
                   textures: bool, grid, share: bool = False) -> tuple | None:
    """prepare_iso() for native.composite_sloped. corner_rise needs no copy:
    SlopedChunkCache replaces it on every refresh, never writes it in place."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    pack = _units_of(pack)
    cx, cy, idx = _candidate_arrays(candidates, w)
    tex = _textures(_tex_ids(scenario, idx, pack), tile_px, textures)
    if tex is None:
        return None
    slots, tex_tables = tex

    nw, ne = corner_rise[cy, cx], corner_rise[cy, cx + 1]
    sw, se = corner_rise[cy + 1, cx], corner_rise[cy + 1, cx + 1]
    dmin = np.minimum(np.minimum(nw, ne), np.minimum(sw, se))
    norm = [nw - dmin, ne - dmin, sw - dmin, se - dmin]
    # One scalar per shape (base past the largest normalized corner), far
    # cheaper to unique than rows.
    base = int(max(int(c.max()) for c in norm)) + 1
    scalar = ((norm[0] * base + norm[1]) * base + norm[2]) * base + norm[3]
    _uniq, first, inv = np.unique(scalar, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    shape_keys = [tuple(int(c[i]) for c in norm) for i in first.tolist()]
    registry = sloped_shapes(tile_px)
    shape_of = np.array(registry.shape_ids(shape_keys), dtype=np.int32)
    cand_shape = np.ascontiguousarray(shape_of[inv], dtype=np.int32)

    shades = [render._slope_shade(tile_px, *key, proj.elev_step) for key in shape_keys]
    shade_start = np.zeros(len(shades), dtype=np.int64)
    np.cumsum([s.size for s in shades[:-1]], out=shade_start[1:])
    shade = np.ascontiguousarray(np.concatenate(shades), dtype=np.float32)
    cand_shade = np.ascontiguousarray(shade_start[inv], dtype=np.int64)

    stamps = None
    cand_grid = np.full(cx.size, -1, dtype=np.int32)
    if grid.paints:
        draw_grid, draw_minors = render._grid_lod_edges_filter(tile_px / 2)
        if draw_grid:
            edge_keys = _edge_states(cx, cy, w, h, draw_minors, None)
            pair = edge_keys * len(shape_keys) + inv
            cand_grid, stamps = _grid_stamps(
                tile_px, grid, pair, split=lambda p: (p // len(shape_keys), shape_keys[p % len(shape_keys)]),
            )
    quads, edges = registry.packed()
    geom = (proj.origin_x, proj.origin_y, proj.half_w, proj.half_h, tile_px)
    return (
        x0, y0, cx, cy, slots, cand_grid, cand_shape, cand_shade, corner_rise, geom,
        quads, edges, registry.diamond, shade, tex_tables, stamps,
        pack.kernel_units(share) if pack is not None else None,
    )
