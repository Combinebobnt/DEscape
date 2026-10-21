"""Terrain blending (TASK-143, GH #13): which neighbours bleed into a tile,
and the blended square block that results. Pure, no Qt. Consumers:
render.py's _top_block() for the blocks, and for the keys the full-map
renderers (influences() directly, the byte-identity oracle) and the chunk
composites (through a KeyMap, which render_cache owns and invalidates).

Two mechanisms, both applied to the square tile_px x tile_px top block every
terrain style builds first:

- Layer ("Layering Mode", TerrainTile.layer): the layer terrain drawn over
  the tile through a terrain/masks/*.png overlay, DE's own shader formula
  tile*(1 - mask.r) + layer*mask.r, the mask cropped like the texture.
- Neighbour transitions: a neighbour of higher blend_priority bleeds into
  the tile through a blendomatic alpha mask. Algorithm, mode table and tile
  numbering from openage, doc/media/blendomatic.md
  (https://github.com/SFTtech/openage/blob/master/doc/media/blendomatic.md).

Polarity, measured on the real blendomatic_x1.dat and pinned by
tests/test_terrain_blend.py: the alpha byte is the tile's OWN weight, the
neighbour's is 128 - alpha. openage's prose says the opposite, but under its
reading corner-only tiles (16-19) would out-cover single-edge ones and tile
30 ("all") would keep its edges and lose its centre.

Accepted approximations: blendomatic shapes, not DE's terrain/blends atlases,
with each bleeding edge faded to 0 own weight as DE's atlases are
(fade_bleeding_edges()) and every other edge conformed to a shared vertex-ramp profile, so two
same-terrain tiles agree along their shared edge (conform_edges()); a neighbour bleeds its base
texture only; the layer blend divides by 256,
not the shader's 255, so a full mask lands up to 1 step short of the layer;
the layer mask is the LAYER
terrain's overlay_mask with LAYER_MASK_FALLBACK for an empty name (both
unconfirmed until the in-game comparison).
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import numpy as np

from descape import asset_source, game_dataset

BLEND_MAP_PATH = Path(__file__).resolve().parent / "terrain_blend_map.json"

# Layer mask for a layer terrain whose overlay_mask is '' (Desert, farms, ...).
LAYER_MASK_FALLBACK = "neutral_50.png"

# Blendomatic mode by (this tile's blend_type, the neighbour's blend_type);
# blend types are land, farm, beach, water, shallows, road, ice, snow.
MODE_TABLE = (
    (2, 3, 2, 1, 1, 6, 5, 4),
    (3, 3, 3, 1, 1, 6, 5, 4),
    (2, 3, 2, 1, 1, 6, 1, 4),
    (1, 1, 1, 0, 7, 6, 5, 4),
    (1, 1, 1, 7, 7, 6, 5, 4),
    (6, 6, 6, 6, 6, 6, 5, 4),
    (5, 5, 1, 5, 5, 5, 5, 4),
    (4, 3, 4, 4, 4, 4, 4, 4),
)

# openage's neighbour bits 0..7 run N, NE, E, SE, S, SW, W, NW on screen. As
# grid offsets (iso_geometry._inverse_sample: +x is the NE edge, +y the SE
# edge), even bits are the diamond's tips and odd bits its edges.
NEIGHBOURS = ((1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1))
EDGE_BITS = 0b10101010
CORNER_BITS = 0b01010101

# Edge-bit combination -> blendomatic tile. A single edge picks one of four
# variants by the low 2 bits of the coordinate running along that edge.
_EDGE_TILE = {
    0b00001000: 0,  # SE
    0b00000010: 4,  # NE
    0b00100000: 8,  # SW
    0b10000000: 12,  # NW
    0b00100010: 20,
    0b10001000: 21,
    0b10100000: 22,
    0b10000010: 23,
    0b00101000: 24,
    0b00001010: 25,
    0b00101010: 26,
    0b10101000: 27,
    0b10100010: 28,
    0b10001010: 29,
    0b10101010: 30,
}
_SINGLE_EDGE_RUNS_ALONG_X = {0b00001000, 0b10000000}
_CORNER_TILE = {0: 18, 2: 16, 4: 17, 6: 19}  # N up, E right, S down, W left

# Blendomatic tile -> the edge bits it bleeds across (corner-only tiles 16-19 touch a point, so none).
_BLEED_EDGES = {
    tile + (k if tile < 16 else 0): edges for edges, tile in _EDGE_TILE.items() for k in range(4 if tile < 16 else 1)
}
# Edge bit -> (square axis, True if that edge is the axis's far end): +y is SE, +x is NE.
_EDGE_SIDE = {3: (0, True), 1: (1, True), 5: (1, False), 7: (0, False)}
# Depth of the edge fade, as a fraction of the tile: two raster rows of the 49-row diamond either side.
EDGE_FADE_DEPTH = 4 / 49


def _edge_ends(edge_bit: int) -> int:
    """The corner bits at the two ends of an edge: edge b runs from corner b - 1 to corner b + 1."""
    return 1 << ((edge_bit - 1) % 8) | 1 << ((edge_bit + 1) % 8)


def _touched(tile: int) -> int:
    edges = _BLEED_EDGES.get(tile, 0)
    out = 0
    for bit in range(1, 8, 2):
        if edges >> bit & 1:
            out |= _edge_ends(bit)
    for bit, corner in _CORNER_TILE.items():
        if corner == tile:
            out |= 1 << bit
    return out


# Blendomatic tile -> the square vertices it bleeds toward, as corner bits (0 N, 2 E, 4 S, 6 W).
_TOUCHED_VERTICES = {tile: _touched(tile) for tile in (*_BLEED_EDGES, *_CORNER_TILE.values())}
# Corner bit -> (on the last row, on the last column) of the square: N top right, E bottom right, ...
_VERTEX_SIDE = {0: (False, True), 2: (True, True), 4: (True, False), 6: (False, False)}
# r(t)'s ramp length, as a fraction of the edge: a non-bleeding edge's own weight rises from 0 at a
# touched vertex to 128 this far from it. At most 1, so an untouched vertex is always 128.
# Measured at 64 px on Dos Pilas and ElCid: every L in {1, 0.7, 0.5} takes the same-terrain mask
# step from max 45.0 summed RGB to 0.0; 0.7 and 0.5 lift mid-edge to 128 and draw own-ground notches
# along cell edges, 1 draws none.
EDGE_RAMP_LENGTH = 1.0
# Depth over which conform_edges() blends each edge's correction into the mask, as a fraction of the tile.
# 8/49 and 12/49 render alike and leave identical edges; 8/49 adds less non-monotonicity inward of
# the edges (352686 vs 376830 summed over every mode 1-2 tile, 312624 unconformed).
EDGE_CONFORM_DEPTH = 8 / 49

# Bounded by bytes, not entries: a 128 px block is 48 KB, an 8 px one 192 B.
BLOCK_CACHE_BYTES = 32 * 1024 * 1024


class BlendEntry(NamedTuple):
    priority: int
    blend_type: int
    overlay_mask: str


def _load_table() -> dict[int, BlendEntry]:
    data = json.loads(game_dataset.table_path(BLEND_MAP_PATH).read_text())
    return {
        int(tid): BlendEntry(e["priority"], e["blend_type"], e["overlay_mask"]) for tid, e in data["terrains"].items()
    }


BLEND_TABLE: dict[int, BlendEntry] = _load_table()


@game_dataset.on_switch
def _reload_table() -> None:
    """Refills BLEND_TABLE in place for the active dataset (GH #67), and drops
    _luts()' copy, which keys on the table's identity and so misses a refill."""
    global _lut_for
    BLEND_TABLE.clear()
    BLEND_TABLE.update(_load_table())
    with _lut_lock:
        _lut_for = None


def tile_indices(bits: int, x: int, y: int) -> tuple[int, ...]:
    """Blendomatic tiles for one neighbour terrain whose influence covers
    the neighbour bits `bits`: the edge-combination tile first (if any edge
    bit is set), then one tile per corner bit. A corner whose adjacent edge
    is already covered by the same terrain adds nothing, since the edge mask
    reaches that tip; that is openage's diagonal rule, per terrain."""
    out = []
    edges = bits & EDGE_BITS
    if edges:
        tile = _EDGE_TILE[edges]
        if tile < 16:
            tile += (x if edges in _SINGLE_EDGE_RUNS_ALONG_X else y) & 3
        out.append(tile)
    for bit, tile in _CORNER_TILE.items():
        if bits >> bit & 1 and not edges & ((1 << ((bit - 1) % 8)) | (1 << (bit + 1))):
            out.append(tile)
    return tuple(out)


def fade_bleeding_edges(alpha: np.ndarray, tile_index: int) -> np.ndarray:
    """A square own-weight mask with its own weight faded to 0 along each
    edge the tile bleeds across, linearly over EDGE_FADE_DEPTH of the tile.
    blendomatic_x1.dat keeps 4-24 of 128 there, which drew a cell-edge step
    against the neighbour (TASK-283); DE's own terrain/blends atlases reach
    0 at the edge. Integer math, so numpy and native share the bytes."""
    edges = _BLEED_EDGES.get(tile_index, 0)
    if not edges:
        return alpha
    tile_px = alpha.shape[0]
    n = max(1, round(tile_px * EDGE_FADE_DEPTH))
    depth = np.arange(tile_px)
    weight = np.full(alpha.shape, n, dtype=np.int32)
    for bit, (axis, far) in _EDGE_SIDE.items():
        if edges >> bit & 1:
            ramp = np.minimum(depth[::-1] if far else depth, n)
            weight = np.minimum(weight, ramp[:, None] if axis == 0 else ramp[None, :])
    return ((alpha.astype(np.int32) * weight + n // 2) // n).astype(np.uint8)


def _ramp(tile_px: int) -> np.ndarray:
    """r at each texel of an edge, counted from a touched vertex: round(128 * min(1, t / L)),
    t = i / (tile_px - 1), so the vertex texel is 0 and the far one 128 at every size."""
    t = np.arange(tile_px) / max(1, tile_px - 1)
    return np.floor(128 * np.minimum(1.0, t / EDGE_RAMP_LENGTH) + 0.5).astype(np.int64)


def edge_profile(tile_index: int, edge_bit: int, tile_px: int) -> np.ndarray:
    """The own weight conform_edges() puts on one non-bleeding edge line, from its low row or
    column end: 128 times r / 128 for each touched end, so it depends on the two vertices only."""
    touched = _TOUCHED_VERTICES.get(tile_index, 0)
    axis = _EDGE_SIDE[edge_bit][0]
    ramp = _ramp(tile_px)
    out = np.full(tile_px, 128, dtype=np.int64)
    for bit in (b for b in _VERTEX_SIDE if _edge_ends(edge_bit) >> b & 1 and touched >> b & 1):
        # The end at the line's high index is the vertex on the last column (row line) or row (column line).
        high = _VERTEX_SIDE[bit][1 - axis]
        out = (out * (ramp[::-1] if high else ramp) + 64) // 128
    return out


def conform_edges(alpha: np.ndarray, tile_index: int) -> np.ndarray:
    """A square own-weight mask with each non-bleeding edge line set to edge_profile(), blended in
    over EDGE_CONFORM_DEPTH of the tile by a Coons-style Boolean sum of the four edge corrections,
    so the interior past that depth keeps the blendomatic shape. Bleeding edges are
    fade_bleeding_edges()' and get no correction. Two same-terrain tiles blended by different
    blendomatic tiles then agree along their shared edge (TASK-284). Integer math, clamped 0..128."""
    tile_px = alpha.shape[0]
    bleeds = _BLEED_EDGES.get(tile_index, 0)
    a = alpha.astype(np.int64)
    lines = {3: np.s_[-1, :], 1: np.s_[:, -1], 5: np.s_[:, 0], 7: np.s_[0, :]}
    deltas = {
        bit: np.zeros(tile_px, dtype=np.int64) if bleeds >> bit & 1 else edge_profile(tile_index, bit, tile_px) - a[line]
        for bit, line in lines.items()
    }
    if not any(d.any() for d in deltas.values()):
        return alpha
    n = min(tile_px - 1, max(1, round(tile_px * EDGE_CONFORM_DEPTH)))
    depth = np.arange(tile_px)
    near = {bit: np.maximum(0, n - (depth[::-1] if far else depth)) for bit, (_axis, far) in _EDGE_SIDE.items()}
    num = np.zeros((tile_px, tile_px), dtype=np.int64)
    for bit, (axis, _far) in _EDGE_SIDE.items():
        num += n * (near[bit][:, None] * deltas[bit][None, :] if axis == 0 else near[bit][None, :] * deltas[bit][:, None])
    touched = _TOUCHED_VERTICES.get(tile_index, 0)
    for vertex, (row_far, col_far) in _VERTEX_SIDE.items():
        ends = [b for b in lines if _edge_ends(b) >> vertex & 1]
        if any(bleeds >> b & 1 for b in ends):
            continue  # a faded edge holds the vertex at 0, which is already its profile value
        row_bit = next(b for b in ends if _EDGE_SIDE[b][0] == 0)
        col_bit = next(b for b in ends if _EDGE_SIDE[b][0] == 1)
        corner = (0 if touched >> vertex & 1 else 128) - a[-1 if row_far else 0, -1 if col_far else 0]
        num -= near[row_bit][:, None] * near[col_bit][None, :] * corner
    out = a + (num + n * n // 2) // (n * n)
    return np.clip(out, 0, 128).astype(np.uint8)


_lut_lock = threading.Lock()
# Holds the table itself, not its id(): a freed table's id can be reused.
_lut_for: tuple[dict, np.ndarray, np.ndarray] | None = None


def _luts() -> tuple[np.ndarray, np.ndarray]:
    """(priority, blend_type) by terrain id, -1 for an id BLEND_TABLE lacks.
    Rebuilt when BLEND_TABLE is rebound (tests swap in a synthetic one)."""
    global _lut_for
    table = BLEND_TABLE
    with _lut_lock:
        if _lut_for is None or _lut_for[0] is not table:
            size = max(table, default=0) + 1
            prio = np.full(size, -1, dtype=np.int64)
            btype = np.full(size, -1, dtype=np.int64)
            for tid, entry in table.items():
                prio[tid] = entry.priority
                btype[tid] = entry.blend_type
            _lut_for = (table, prio, btype)
        return _lut_for[1], _lut_for[2]


def _lookup(lut: np.ndarray, tids: np.ndarray) -> np.ndarray:
    known = (tids >= 0) & (tids < lut.size)
    return np.where(known, lut[np.where(known, tids, 0)], -1)


def influences(tid_mirror: np.ndarray, layer_mirror: np.ndarray, xs, ys) -> list:
    """Per candidate tile (xs[i], ys[i]), its blend key, or None for a tile
    with no layer and no higher-priority neighbour, which keeps today's
    plain crop. Both mirrors are 2-D (map_h, map_w) terrain-id grids indexed
    [y, x] (render_cache's flat y*w + x mirror needs .reshape(h, w)), the
    layer one signed, holding -1 where a tile has no layer. tid_mirror must
    hold raw terrain ids, never a farm override; skipping farm-overridden
    candidates is the caller's job. Candidates must lie on the map.

    A key is (own tid, layer tid or -1, ((n_tid, mode, tile_idx), ...)),
    hashable, the influences in ascending neighbour priority, each
    neighbour terrain's tiles in tile_indices() order. A layer equal to the
    tile's own terrain is a no-op and reads as -1."""
    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    if xs.size == 0:
        return []
    prio_lut, type_lut = _luts()
    if tid_mirror.ndim != 2 or layer_mirror.shape != tid_mirror.shape:
        raise ValueError(f"mirrors must be matching 2-D grids, got {tid_mirror.shape} and {layer_mirror.shape}")
    h, w = tid_mirror.shape
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= w or ys.max() >= h:
        raise ValueError("candidate tile off the map")
    own = tid_mirror[ys, xs].astype(np.int64)
    layer = layer_mirror[ys, xs].astype(np.int64)
    layer = np.where(layer == own, -1, layer)
    own_prio = _lookup(prio_lut, own)

    n_tids = np.empty((8, xs.size), dtype=np.int64)
    for bit, (dx, dy) in enumerate(NEIGHBOURS):
        nx, ny = xs + dx, ys + dy
        inside = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
        n_tids[bit] = np.where(inside, tid_mirror[np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)], -1)
    n_prio = _lookup(prio_lut, n_tids)
    bleeds = (n_prio > own_prio) & (own_prio >= 0) & (n_tids != own)

    keys: list = [None] * xs.size
    for i in np.flatnonzero(bleeds.any(axis=0) | (layer >= 0)).tolist():
        own_tid = int(own[i])
        by_terrain: dict[int, int] = {}
        for bit in np.flatnonzero(bleeds[:, i]).tolist():
            n_tid = int(n_tids[bit, i])
            by_terrain[n_tid] = by_terrain.get(n_tid, 0) | (1 << bit)
        entries = []
        for n_tid in sorted(by_terrain, key=lambda t: int(prio_lut[t])):
            mode = MODE_TABLE[int(type_lut[own_tid])][int(type_lut[n_tid])]
            entries.extend((n_tid, mode, tile) for tile in tile_indices(by_terrain[n_tid], int(xs[i]), int(ys[i])))
        keys[i] = (own_tid, int(layer[i]), tuple(entries))
    return keys


# KeyMap.ids values that are not a spec id: no key, and not derived yet.
NO_KEY = -1
UNKNOWN = -2
# Past this many interned keys a KeyMap starts over (lookup()).
KEY_MAP_MAX_SPECS = 1 << 16
# Signatures a KeyMap's stash keeps per tile, FIFO (TASK-031.69 step 0: 3 hit 99.94% at undo 1).
KEY_STASH_SLOTS = 3
# A tile and its 8 neighbours, as (dy, dx) offsets.
_RING_DY, _RING_DX = (a.ravel() for a in np.mgrid[-1:2, -1:2])
# The 8 neighbours in influences()' order, as (dx, dy) offsets.
_NB_DX, _NB_DY = (np.array(a, dtype=np.int64) for a in zip(*NEIGHBOURS, strict=True))
# A stash signature: own tid, layer and the 8 neighbour tids, each as v + 1 in 12 bits, five per word.
_SIG_SHIFTS = np.arange(5, dtype=np.int64) * 12
_SIG_MAX = (1 << 12) - 2


class KeyMap:
    """influences() per tile of one map, memoized: `ids` holds a spec id per
    tile index (y * map_w + x), NO_KEY, or UNKNOWN until a lookup() derives
    it. A spec id indexes `keys` (the influences() key), `own` (its own tid)
    and `extra` (the sorted layer and neighbour tids it names).

    The map equals influences() over its sources only while every write to
    them is followed by invalidate() of the tiles whose terrain id or layer
    changed, and every replaced source by bind(). Not thread-safe: one
    thread reads and writes it (the GUI thread, for a chunk cache's).

    With mirrors bound (bind(..., mirrors=)), a stash keeps the spec id of
    the last KEY_STASH_SLOTS exact neighbourhoods each tile was derived in
    (own tid, layer, 8 neighbour tids; off-map -1), and lookup() takes an
    UNKNOWN tile's id from it on an exact match instead of deriving, so an
    undo re-uses the keys its terrain had. That is exact only while a key
    is a function of those 10 values, the tile's position and the bound
    sources: a new input to influences() must reset the map when it
    changes. invalidate() keeps the stash; reset() clears it."""

    def __init__(self, map_w: int, map_h: int):
        self.w, self.h = map_w, map_h
        self._sources: tuple = ()
        self._mirrors: tuple | None = None
        # (signatures (n, K, 2) int64, spec ids (n, K) int32, next slot (n,) uint8), on the first mirrored bind.
        self._stash: tuple | None = None
        self.reset()

    def reset(self) -> None:
        """Forgets every key, the stash's included. Spec ids handed out
        before this are void."""
        self.ids = np.full(self.w * self.h, UNKNOWN, dtype=np.int32)
        self.keys: list = []
        self.own = np.zeros(64, dtype=np.int64)
        self.extra: list[tuple[int, ...]] = []
        self._id_of: dict = {}
        if self._stash is not None:
            sig, sid, nxt = self._stash
            sig.fill(-1)  # no packed signature is negative, so an empty slot never matches
            sid.fill(UNKNOWN)
            nxt.fill(0)

    def bind(self, sources: tuple, mirrors: tuple | None = None) -> None:
        """Names what the keys were derived from. The objects are held and
        compared by identity (never id(), which a freed object's can reuse);
        any element replaced resets the map. An in-place change to one is the
        caller's to invalidate().

        mirrors: the (terrain id, layer) grids, flat or 2-D, each an element
        of `sources`, turns the stash on (class docstring). lookup()'s derive
        must then be influences() over exactly those two."""
        if len(sources) != len(self._sources) or any(a is not b for a, b in zip(sources, self._sources, strict=True)):
            self.reset()
            self._sources = sources
        if mirrors is not None:
            n = self.w * self.h
            assert len(mirrors) == 2 and all(m.size == n for m in mirrors), "mirrors sized for another map"
            assert all(any(m is s for s in sources) for m in mirrors), "a mirror must be a bound source"
            if self._stash is None:
                k = KEY_STASH_SLOTS
                self._stash = (
                    np.full((n, k, 2), -1, dtype=np.int64),
                    np.full((n, k), UNKNOWN, dtype=np.int32),
                    np.zeros(n, dtype=np.uint8),
                )
        self._mirrors = mirrors

    def _intern(self, key) -> int:
        sid = self._id_of.get(key)
        if sid is None:
            sid = self._id_of[key] = len(self.keys)
            self.keys.append(key)
            if sid >= self.own.size:
                self.own = np.concatenate([self.own, np.zeros_like(self.own)])
            self.own[sid] = key[0]
            named = {entry[0] for entry in key[2]}
            if key[1] >= 0:
                named.add(key[1])
            self.extra.append(tuple(sorted(named)))
        return sid

    def lookup(self, idx: np.ndarray, derive) -> np.ndarray:
        """The spec id (or NO_KEY) of each tile index in idx, int32.
        derive(xs, ys) returns the influences() keys of the tiles this map
        has not derived yet. Resets first once past KEY_MAP_MAX_SPECS, so no
        id read in this call outlives its table."""
        if len(self.keys) > KEY_MAP_MAX_SPECS:
            self.reset()
        ids = self.ids[idx]
        missing = ids == UNKNOWN
        if missing.any():
            miss = np.unique(idx[missing])
            if self._mirrors is None or self._stash is None:
                ys, xs = np.divmod(miss, self.w)
                intern = self._intern
                self.ids[miss] = [NO_KEY if key is None else intern(key) for key in derive(xs, ys)]
            else:
                self._lookup_stashed(miss, derive)
            ids = self.ids[idx]
        return ids

    def _signatures(self, miss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(signature (m, 2) int64, stashable (m,) bool) of each tile index in
        miss, read from the bound mirrors now. A tile with any of its 10
        values outside [-1, _SIG_MAX] is not stashable."""
        tids = self._mirrors[0].reshape(-1)
        layers = self._mirrors[1].reshape(-1)
        w, h = self.w, self.h
        ys, xs = np.divmod(miss, w)
        nx = xs[:, None] + _NB_DX
        ny = ys[:, None] + _NB_DY
        inside = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
        vals = np.empty((miss.size, 10), dtype=np.int64)
        vals[:, 0] = tids[miss]
        vals[:, 1] = layers[miss]
        vals[:, 2:] = np.where(inside, tids[np.where(inside, ny * w + nx, 0)], -1)
        vals += 1
        ok = ((vals >= 0) & (vals <= _SIG_MAX + 1)).all(axis=1)
        vals &= 0xFFF
        sig = np.bitwise_or.reduce(vals.reshape(-1, 2, 5) << _SIG_SHIFTS, axis=2)
        return sig, ok

    def _lookup_stashed(self, miss: np.ndarray, derive) -> None:
        """lookup()'s fill of the UNKNOWN tile indices `miss` (unique) with
        the stash on: an exact signature match takes its spec id, the rest
        are derived and stored in their tile's next FIFO slot."""
        st_sig, st_sid, st_next = self._stash
        sig, ok = self._signatures(miss)
        match = (st_sig[miss] == sig[:, None, :]).all(axis=2)
        hit = ok & match.any(axis=1)
        if hit.any():
            tiles = miss[hit]
            self.ids[tiles] = st_sid[tiles, match[hit].argmax(axis=1)]
        rest = ~hit
        if not rest.any():
            return
        tiles = miss[rest]
        ys, xs = np.divmod(tiles, self.w)
        intern = self._intern
        new = np.array([NO_KEY if key is None else intern(key) for key in derive(xs, ys)], dtype=np.int32)
        self.ids[tiles] = new
        keep = ok[rest]
        tiles = tiles[keep]
        slot = st_next[tiles]
        st_sig[tiles, slot] = sig[rest][keep]
        st_sid[tiles, slot] = new[keep]
        st_next[tiles] = (slot + 1) % st_sid.shape[1]

    def invalidate(self, idx) -> None:
        """Marks each tile index in idx and its 8 neighbours UNKNOWN, clipped
        to the map. A key reads its own tid and layer and the 8 neighbours'
        tids, so the ring is exact for a tid change and conservative for a
        layer-only one."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return
        ys, xs = np.divmod(idx, self.w)
        ny = ys[:, None] + _RING_DY
        nx = xs[:, None] + _RING_DX
        inside = (nx >= 0) & (nx < self.w) & (ny >= 0) & (ny < self.h)
        self.ids[(ny * self.w + nx)[inside]] = UNKNOWN


class RectKeys(NamedTuple):
    """One composite's blend keys: a KeyMap spec id (or NO_KEY) per
    candidate, and the KeyMap they index. Valid until that map's next
    lookup(), which may reset it."""

    ids: np.ndarray
    table: KeyMap

    def as_list(self) -> list:
        """The influences() key or None per candidate, for the per-tile paths."""
        keys = self.table.keys
        return [keys[i] if i >= 0 else None for i in self.ids.tolist()]


def layer_mask_name(layer_tid: int) -> str:
    entry = BLEND_TABLE.get(layer_tid)
    return entry.overlay_mask if entry is not None and entry.overlay_mask else LAYER_MASK_FALLBACK


_cache_lock = threading.Lock()
_blocks: OrderedDict = OrderedDict()
_block_bytes = 0
# Bumped by clear_caches(), so a block composed from the old install is never inserted.
_generation = 0


def clear_caches() -> None:
    """Drops every cached block. asset_source.clear_install_caches() calls
    this, since the blocks are built from its textures and masks."""
    global _block_bytes, _generation
    with _cache_lock:
        _blocks.clear()
        _block_bytes = 0
        _generation += 1


def _crop(arr: np.ndarray, ox: int, oy: int, tile_px: int) -> np.ndarray:
    return arr[oy : oy + tile_px, ox : ox + tile_px]


def _compose(key, ox: int, oy: int, tile_px: int) -> np.ndarray | None:
    own_tid, layer_tid, entries = key
    size = asset_source.terrain_texture_size(tile_px)
    base = asset_source.terrain_texture_at(own_tid, size)
    if base is None:
        return None
    out = _crop(base, ox, oy, tile_px).astype(np.uint32)
    if layer_tid >= 0:
        layer = asset_source.terrain_texture_at(layer_tid, size)
        mask = asset_source.terrain_mask_at(layer_mask_name(layer_tid), size)
        if layer is not None and mask is not None:
            m = _crop(mask, ox, oy, tile_px).astype(np.uint32)[:, :, None]
            out = (out * (256 - m) + _crop(layer, ox, oy, tile_px) * m + 128) >> 8
    for n_tid, mode, tile in entries:
        neighbour = asset_source.terrain_texture_at(n_tid, size)
        alpha = asset_source.get_blend_mask(mode, tile, tile_px)
        if neighbour is None or alpha is None:
            continue
        a = alpha.astype(np.uint32)[:, :, None]
        out = (out * a + _crop(neighbour, ox, oy, tile_px) * (128 - a) + 64) >> 7
    block = out.astype(np.uint8)
    block.flags.writeable = False
    return block


def blended_block(key, ox: int, oy: int, tile_px: int) -> np.ndarray | None:
    """The (tile_px, tile_px, 3) uint8 square top block for a non-None
    influences() key, read-only and cached: the own texture cropped at
    (ox, oy) (render._crop_offset's), then the layer
    (base*(256 - m) + layer*m + 128) >> 8 with m the 0..255 mask, then each
    influence (dst*a + n*(128 - a) + 64) >> 7 with a the blendomatic
    own-weight alpha. Every texture and mask is cropped at THIS tile's
    (ox, oy), so a bleed continues the neighbour's own tiles. Integer math
    only, so numpy and native can share the bytes. A missing layer,
    neighbour texture or mask is skipped; None when the own texture is
    missing (the caller's flat-colour fallback)."""
    cache_key = (key, ox, oy, tile_px)
    with _cache_lock:
        block = _blocks.get(cache_key)
        if block is not None:
            _blocks.move_to_end(cache_key)
            return block
        generation = _generation
    block = _compose(key, ox, oy, tile_px)
    if block is None:
        return None
    global _block_bytes
    with _cache_lock:
        if generation == _generation and cache_key not in _blocks:
            _blocks[cache_key] = block
            _block_bytes += block.nbytes
            while _block_bytes > BLOCK_CACHE_BYTES and len(_blocks) > 1:
                _, evicted = _blocks.popitem(last=False)
                _block_bytes -= evicted.nbytes
    return block
