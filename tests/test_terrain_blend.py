"""TASK-143 step 2: terrain_blend's pure core.

The real-asset tests settle the blendomatic alpha's polarity and pin the
tile numbering against the real file (skip without AOE2DE_INSTALL_PATH).
The synthetic tests (pattern: test_native_composite.py's synthetic_textures,
plus a synthetic blend table and blendomatic) need no install.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from descape import asset_source, render, terrain_blend
from descape.terrain_blend import NEIGHBOURS, BlendEntry, influences, tile_indices

TEXTURE_SIZE = 512
LOW, HIGH, HIGHER, LAYER = 1, 2, 3, 4


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")


def _raster_uv() -> tuple[np.ndarray, np.ndarray]:
    rows_n, width = asset_source.BLENDOMATIC_ROWS, asset_source.BLENDOMATIC_WIDTH
    widths = [4 * min(r, rows_n - 1 - r) + 1 for r in range(rows_n)]
    rows = np.concatenate([np.full(w, r) for r, w in enumerate(widths)])
    cols = np.concatenate([np.arange(w) + (width - w) // 2 for w in widths])
    return (cols + 0.5) / width * 2 - 1, (rows + 0.5) / rows_n * 2 - 1


def _bit(dx: int, dy: int) -> int:
    return NEIGHBOURS.index((dx, dy))


# --- polarity and orientation on the real blendomatic ---

SINGLE_EDGE = range(16)
CORNER_ONLY = range(16, 20)
TWO_ADJACENT_EDGES = range(22, 26)


def _coverage(mode: int = 2) -> np.ndarray:
    """Neighbour weight per pixel, 128 - alpha: alpha is the tile's own."""
    return 128 - asset_source.get_blendomatic()[mode].astype(np.int64)


def test_real_alpha_is_the_tiles_own_weight():
    """Measured in mode 2: neighbour coverage orders corner-only (21-27)
    < single-edge (64-72) < two adjacent edges (106-112). Flipped, corner-only
    would cover most. Tiles 20, 21 and 30 keep a centre band or blob, so
    they cover about as much as a single edge and are checked by shape."""
    _require_install()
    mean = _coverage().mean(axis=1)
    assert mean[list(CORNER_ONLY)].max() < mean[list(SINGLE_EDGE)].min()
    assert mean[list(SINGLE_EDGE)].max() < mean[list(TWO_ADJACENT_EDGES)].min()
    u, v = _raster_uv()
    ring, centre = np.abs(u) + np.abs(v) > 0.85, np.abs(u) + np.abs(v) < 0.3
    cov_all = _coverage()[30]
    assert cov_all[ring].mean() > 100 and cov_all[centre].mean() < 30, "tile 30 should give the neighbour every edge"


def test_real_tile_numbering_matches_openage_on_screen():
    """Single-edge and corner tiles' coverage centroids point where
    openage's blendomatic.md names them: 0-3 lower right, 4-7 upper right,
    8-11 lower left, 12-15 upper left, 16 right, 17 down, 18 up, 19 left."""
    _require_install()
    u, v = _raster_uv()
    cov = _coverage()
    screen = {0: (1, 1), 4: (1, -1), 8: (-1, 1), 12: (-1, -1), 16: (1, 0), 17: (0, 1), 18: (0, -1), 19: (-1, 0)}
    for first, (su, sv) in screen.items():
        for tile in range(first, first + 4) if first < 16 else (first,):
            cu = (cov[tile] * u).sum() / cov[tile].sum()
            cv = (cov[tile] * v).sum() / cov[tile].sum()
            for want, got in ((su, cu), (sv, cv)):
                if want:
                    assert np.sign(got) == want and abs(got) > 0.1, (tile, cu, cv)
                else:
                    assert abs(got) < 0.05, (tile, cu, cv)


@pytest.mark.parametrize("bit", range(8))
def test_real_square_masks_bleed_from_the_neighbours_grid_side(bit):
    """End to end in DEscape's grid: the tile tile_indices() picks for one
    neighbour, resampled to the square block, has its coverage centroid on
    that neighbour's side (s along +x, t along +y)."""
    _require_install()
    dx, dy = NEIGHBOURS[bit]
    (tile,) = tile_indices(1 << bit, 0, 0)
    tile_px = 64
    cov = 128 - asset_source.get_blend_mask(2, tile, tile_px).astype(np.int64)
    t, s = (np.mgrid[0:tile_px, 0:tile_px] + 0.5) / tile_px
    cs = (cov * s).sum() / cov.sum() - 0.5
    ct = (cov * t).sum() / cov.sum() - 0.5
    for want, got in ((dx, cs), (dy, ct)):
        if want:
            assert np.sign(got) == want and abs(got) > 0.08, (bit, tile, cs, ct)
        else:
            assert abs(got) < 0.05, (bit, tile, cs, ct)


@pytest.mark.parametrize("edges", [e for e in range(256) if e and not e & terrain_blend.CORNER_BITS])
def test_real_edge_combination_tiles_lean_toward_their_covered_edges(edges):
    """Every edge combination's tile, resampled to the square block, has its
    coverage centroid on the side the covered edges' offsets sum to. Opposite
    pairs and all four sum to zero and must stay central."""
    _require_install()
    (tile,) = tile_indices(edges, 0, 0)
    bits = [b for b in range(8) if edges >> b & 1]
    sx, sy = sum(NEIGHBOURS[b][0] for b in bits), sum(NEIGHBOURS[b][1] for b in bits)
    tile_px = 64
    cov = 128 - asset_source.get_blend_mask(2, tile, tile_px).astype(np.int64)
    t, s = (np.mgrid[0:tile_px, 0:tile_px] + 0.5) / tile_px
    cs = (cov * s).sum() / cov.sum() - 0.5
    ct = (cov * t).sum() / cov.sum() - 0.5
    for want, got in ((sx, cs), (sy, ct)):
        if want:
            assert np.sign(got) == np.sign(want) and abs(got) > 0.02, (edges, tile, cs, ct)
        else:
            assert abs(got) < 0.03, (edges, tile, cs, ct)


# --- tile_indices ---


def test_single_edges_pick_a_variant_along_the_edge():
    se, ne = 1 << _bit(0, 1), 1 << _bit(1, 0)
    assert tile_indices(se, 5, 0) == (1,)  # SE runs along x
    assert tile_indices(se, 5, 3) == (1,)
    assert tile_indices(ne, 5, 3) == (7,)  # NE runs along y
    assert tile_indices(1 << _bit(-1, 0), 0, 2) == (10,)
    assert tile_indices(1 << _bit(0, -1), 6, 0) == (14,)


def test_a_corner_beside_a_covered_edge_adds_nothing():
    ne, n, e, s = (1 << _bit(*d) for d in ((1, 0), (1, -1), (1, 1), (-1, 1)))
    assert tile_indices(ne | n | e, 0, 0) == (4,)
    assert tile_indices(ne | s, 0, 0) == (4, 17)
    assert tile_indices(n | e | s, 0, 0) == (18, 16, 17)


def test_edge_combinations_follow_openage():
    ne, se, sw, nw = (1 << _bit(*d) for d in ((1, 0), (0, 1), (-1, 0), (0, -1)))
    assert tile_indices(ne | sw, 0, 0) == (20,)
    assert tile_indices(se | nw, 0, 0) == (21,)
    assert tile_indices(ne | se, 0, 0) == (25,)
    assert tile_indices(ne | se | sw | nw, 0, 0) == (30,)


# --- conform_edges (TASK-284), synthetic masks ---

# Hand-typed oracle: the square vertices each blendomatic tile bleeds toward, and where they sit.
_TOUCHED = {
    **dict.fromkeys(range(4), "ES"),
    **dict.fromkeys(range(4, 8), "NE"),
    **dict.fromkeys(range(8, 12), "SW"),
    **dict.fromkeys(range(12, 16), "WN"),
    16: "E",
    17: "S",
    18: "N",
    19: "W",
    20: "NESW",
    21: "NESW",
    22: "SWN",
    23: "WNE",
    24: "ESW",
    25: "NES",
    **dict.fromkeys(range(26, 31), "NESW"),
}
_BLEEDS = {
    **dict.fromkeys(range(4), ("SE",)),
    **dict.fromkeys(range(4, 8), ("NE",)),
    **dict.fromkeys(range(8, 12), ("SW",)),
    **dict.fromkeys(range(12, 16), ("NW",)),
    **dict.fromkeys(range(16, 20), ()),
    20: ("NE", "SW"),
    21: ("SE", "NW"),
    22: ("SW", "NW"),
    23: ("NW", "NE"),
    24: ("SE", "SW"),
    25: ("NE", "SE"),
    26: ("NE", "SE", "SW"),
    27: ("SE", "SW", "NW"),
    28: ("NE", "SW", "NW"),
    29: ("NE", "SE", "NW"),
    30: ("NE", "SE", "SW", "NW"),
}
# Edge -> (its texel line, the vertex at the line's index 0, the vertex at its last index).
_EDGES = {
    "NW": (np.s_[0, :], "W", "N"),
    "NE": (np.s_[:, -1], "N", "E"),
    "SE": (np.s_[-1, :], "S", "E"),
    "SW": (np.s_[:, 0], "W", "S"),
}


def _oracle_profile(tile: int, edge: str, tile_px: int) -> np.ndarray:
    """The plan's canonical profile: 128 times r(distance from v) / 128 per touched end v."""
    _line, lo, hi = _EDGES[edge]
    i = np.arange(tile_px)
    out = np.full(tile_px, 128, dtype=np.int64)
    for vertex, dist in ((lo, i), (hi, tile_px - 1 - i)):
        if vertex in _TOUCHED[tile]:
            r = np.floor(128 * np.minimum(1.0, dist / (tile_px - 1) / terrain_blend.EDGE_RAMP_LENGTH) + 0.5)
            out = (out * r.astype(np.int64) + 64) // 128
    return out


def _conformed(mask: np.ndarray, tile: int) -> np.ndarray:
    return terrain_blend.conform_edges(terrain_blend.fade_bleeding_edges(mask, tile), tile)


@pytest.mark.parametrize("tile_px", (8, 16, 64, 128))
def test_conformed_edges_follow_the_vertex_profile_and_spare_the_interior(tile_px):
    """Every tile 0-30, on random masks: each non-bleeding edge line is exactly the canonical
    profile, each bleeding one is 0, everything past EDGE_CONFORM_DEPTH is untouched, 0..128."""
    rng = np.random.default_rng(tile_px)
    n = min(tile_px - 1, max(1, round(tile_px * terrain_blend.EDGE_CONFORM_DEPTH)))
    for tile in range(31):
        mask = rng.integers(0, 129, (tile_px, tile_px)).astype(np.uint8)
        faded = terrain_blend.fade_bleeding_edges(mask, tile)
        out = terrain_blend.conform_edges(faded, tile)
        assert out.dtype == np.uint8 and int(out.max()) <= 128
        for edge, (line, _lo, _hi) in _EDGES.items():
            if edge in _BLEEDS[tile]:
                assert not out[line].any(), (tile, edge)
            else:
                assert np.array_equal(out[line], _oracle_profile(tile, edge, tile_px)), (tile, edge, out[line])
        assert np.array_equal(out[n:-n, n:-n], faded[n:-n, n:-n]), tile


def test_the_touched_vertex_table_matches_the_hand_oracle():
    name = {0: "N", 2: "E", 4: "S", 6: "W"}
    for tile in range(31):
        got = {name[b] for b in name if terrain_blend._TOUCHED_VERTICES[tile] >> b & 1}
        assert got == set(_TOUCHED[tile]), tile


_OWN, _ABOVE, _HIGHER, _BELOW = 1, 2, 3, 4
_COLOURS = {_OWN: (200, 40, 90), _ABOVE: (10, 230, 120), _HIGHER: (90, 0, 255), _BELOW: (255, 255, 0)}


@pytest.fixture
def seam_masks(monkeypatch):
    """Constant-colour textures and a random 0..128 mask per (mode, tile), served by
    get_blend_mask as the loader does: faded, then conform_edges, looked up at call time."""
    size = asset_source.LOADED_TEXTURE_SIZE
    textures = {tid: np.broadcast_to(np.array(c, dtype=np.uint8), (size, size, 3)) for tid, c in _COLOURS.items()}
    def texture(tid):
        return textures.get(tid)

    monkeypatch.setattr(asset_source, "get_terrain_texture_array", _fake(texture))
    monkeypatch.setattr(
        terrain_blend,
        "BLEND_TABLE",
        {
            _BELOW: BlendEntry(5, 0, ""),
            _OWN: BlendEntry(10, 0, ""),
            _ABOVE: BlendEntry(20, 3, ""),
            _HIGHER: BlendEntry(30, 5, ""),
        },
    )
    rng = np.random.default_rng(20261011)
    raw: dict = {}

    def mask(mode, tile, tile_px):
        if (mode, tile, tile_px) not in raw:
            raw[mode, tile, tile_px] = rng.integers(0, 129, (tile_px, tile_px)).astype(np.uint8)
        return _conformed(raw[mode, tile, tile_px], tile)

    monkeypatch.setattr(asset_source, "get_blend_mask", _fake(mask))
    asset_source.clear_install_caches()


# Orientation -> (grid w, h, A, B, A's line on the shared edge, B's line on it).
_SEAM_ORIENTATIONS = {
    "SE/NW": (3, 4, (1, 1), (1, 2), np.s_[-1, :], np.s_[0, :]),
    "NE/SW": (4, 3, (1, 1), (2, 1), np.s_[:, -1], np.s_[:, 0]),
}


def _seam_steps(orientation: str, fills, tile_px: int) -> list[tuple[int, int, tuple]]:
    """Per fill of the 10 cells around A and B: (max channel step between A's and B's shared-edge
    lines, the rounding allowance of 1 per influence entry on either side, the grid)."""
    w, h, a, b, a_line, b_line = _SEAM_ORIENTATIONS[orientation]
    others = [(x, y) for y in range(h) for x in range(w) if (x, y) not in (a, b)]
    layers = np.full((h, w), -1, dtype=np.int64)
    blocks: dict = {}

    def block(key):
        if key is None:
            return np.broadcast_to(np.array(_COLOURS[_OWN], dtype=np.int64), (tile_px, tile_px, 3))
        if key not in blocks:
            blocks[key] = terrain_blend._compose(key, 0, 0, tile_px).astype(np.int64)
        return blocks[key]

    out = []
    for fill in fills:
        tids = np.full((h, w), _OWN, dtype=np.int64)
        for (x, y), tid in zip(others, fill, strict=True):
            tids[y, x] = tid
        key_a, key_b = influences(tids, layers, [a[0], b[0]], [a[1], b[1]])
        step = int(np.abs(block(key_a)[a_line] - block(key_b)[b_line]).max())
        allowance = max(len(k[2]) if k else 0 for k in (key_a, key_b))
        out.append((step, allowance, tuple(map(tuple, tids))))
    return out


def _all_fills(orientation: str, tile_px: int) -> list:
    import itertools

    exhaustive = _seam_steps(orientation, itertools.product((_OWN, _ABOVE), repeat=10), tile_px)
    rng = np.random.default_rng(284)
    mixed = rng.choice((_OWN, _ABOVE, _HIGHER, _BELOW), size=(600, 10))
    return exhaustive + _seam_steps(orientation, mixed.tolist(), tile_px)


@pytest.mark.parametrize("tile_px", (8, 32))
@pytest.mark.parametrize("orientation", tuple(_SEAM_ORIENTATIONS))
def test_same_terrain_neighbours_agree_along_their_shared_edge(seam_masks, orientation, tile_px):
    """TASK-284's invariant. Two same-terrain cells A, B, their 10 surrounding cells drawn from
    {own, T} exhaustively (1024), then a seeded sample with a terrain below A and two above it.
    A's and B's lines on their shared edge agree within 1 per influence entry per channel."""
    steps = _all_fills(orientation, tile_px)
    keyed = [s for s in steps if s[1]]
    assert len(keyed) > 1000, "too few keyed pairs to mean anything"
    worst = max(steps, key=lambda s: s[0] - s[1])
    assert worst[0] <= worst[1], worst


def test_the_seam_agreement_catches_unconformed_masks(seam_masks, monkeypatch):
    """Control: with conform_edges patched to identity (gen_review_pack's seam-conform-off
    inject), the same pairs step by far more than their rounding allowance."""
    monkeypatch.setattr(terrain_blend, "conform_edges", lambda alpha, _tile: alpha)
    steps = _all_fills("SE/NW", 32)
    assert max(s[0] - s[1] for s in steps) > 20


# --- synthetic fixtures ---


def _texture(terrain_id: int) -> np.ndarray:
    y, x = np.mgrid[0:TEXTURE_SIZE, 0:TEXTURE_SIZE]
    t = terrain_id + 1
    return np.stack(
        [(x * 7 + y * 13 + t * 31) % 256, (x * y + t * 17) % 256, (x ^ (y * t)) % 256], axis=-1
    ).astype(np.uint8)


def _fake(fn):
    fn.cache_clear = lambda: None
    return fn


@pytest.fixture
def synthetic(monkeypatch):
    """Synthetic textures, blend table and blendomatic. Each single-edge and
    corner tile gives the neighbour (alpha 0) the raster pixels on its own
    screen side and keeps the rest (alpha 128); tiles 20-30 keep everything."""
    textures = {tid: _texture(tid) for tid in (LOW, HIGH, HIGHER, LAYER)}
    masks: dict[str, np.ndarray] = {}

    def texture(tid):
        return textures.get(tid)

    def mask(name):
        return masks.get(name)

    monkeypatch.setattr(asset_source, "get_terrain_texture_array", _fake(texture))
    monkeypatch.setattr(
        terrain_blend,
        "BLEND_TABLE",
        {
            LOW: BlendEntry(10, 0, "low.png"),
            HIGH: BlendEntry(20, 3, ""),
            HIGHER: BlendEntry(30, 2, "higher.png"),
            LAYER: BlendEntry(5, 0, "layer.png"),
        },
    )
    u, v = _raster_uv()
    tiles = np.full((9, 31, asset_source.BLENDOMATIC_TILE_SIZE), 128, dtype=np.uint8)
    screen = {0: (1, 1), 4: (1, -1), 8: (-1, 1), 12: (-1, -1), 16: (1, 0), 17: (0, 1), 18: (0, -1), 19: (-1, 0)}
    for first, (su, sv) in screen.items():
        for tile in range(first, first + 4) if first < 16 else (first,):
            tiles[:, tile][:, su * u + sv * v > 0.2] = 0
    monkeypatch.setattr(asset_source, "get_blendomatic", _fake(lambda: tiles))
    monkeypatch.setattr(asset_source, "get_terrain_mask_array", _fake(mask))
    asset_source.clear_install_caches()
    return SimpleNamespace(textures=textures, masks=masks)


def _grid(w: int = 8, h: int = 8, fill: int = LOW) -> tuple[np.ndarray, np.ndarray]:
    return np.full((h, w), fill, dtype=np.int64), np.full((h, w), -1, dtype=np.int64)


def _plain(tid: int, x: int, y: int, tile_px: int) -> np.ndarray:
    return render._tile_block(SimpleNamespace(x=x, y=y, terrain_id=tid), tile_px)


def _offsets(x: int, y: int, tile_px: int) -> tuple[int, int]:
    return render._crop_offset(x, y, TEXTURE_SIZE, tile_px)


# --- influences ---


def test_an_interior_unlayered_tile_has_no_key_and_keeps_todays_crop(synthetic):
    tids, layers = _grid()
    assert influences(tids, layers, [3, 0, 7], [3, 0, 7]) == [None, None, None]
    tile_px = 32
    ox, oy = _offsets(3, 3, tile_px)
    block = terrain_blend.blended_block((LOW, -1, ()), ox, oy, tile_px)
    assert np.array_equal(block, _plain(LOW, 3, 3, tile_px))


def test_only_the_lower_priority_side_of_a_boundary_gets_a_key(synthetic):
    tids, layers = _grid()
    tids[:, 5:] = HIGH
    low, high = influences(tids, layers, [4, 5], [2, 2])
    assert high is None
    mode = terrain_blend.MODE_TABLE[0][3]
    assert low == (LOW, -1, ((HIGH, mode, 4 + (2 & 3)),))


def test_influences_apply_in_ascending_priority_and_skip_unknown_and_off_map(synthetic):
    tids, layers = _grid(4, 4)
    tids[1, 2] = HIGHER  # NE of (1, 1)
    tids[2, 1] = HIGH  # SE of (1, 1)
    tids[0, 1] = 99  # NW of (1, 1): not in the table
    (key,) = influences(tids, layers, [1], [1])
    assert [entry[0] for entry in key[2]] == [HIGH, HIGHER]
    assert influences(tids, layers, [3], [3]) == [None]  # off-map neighbours on two sides


def test_a_layer_gives_a_key_unless_it_is_the_tiles_own_terrain(synthetic):
    tids, layers = _grid()
    layers[2, 2] = LAYER
    layers[3, 3] = LOW
    assert influences(tids, layers, [2, 3], [2, 3]) == [(LOW, LAYER, ()), None]


# --- blended_block ---


@pytest.mark.parametrize("tile_px", (8, 16, 32, 64, 128))
def test_a_bleed_changes_only_the_lower_tiles_neighbour_side(synthetic, tile_px):
    """Away from the two edges conform_edges() ramps (TASK-284: a lone NE bleed now runs the NW and
    SE edge lines from 0 at its vertex to 128 at the far one), only the +x half changes; the change
    as a whole leans +x; and wherever the served mask is 0 the block is the neighbour's own crop."""
    tids, layers = _grid()
    tids[:, 5:] = HIGH
    (key,) = influences(tids, layers, [4], [2])
    ox, oy = _offsets(4, 2, tile_px)
    block = terrain_blend.blended_block(key, ox, oy, tile_px)
    changed = np.any(block != _plain(LOW, 4, 2, tile_px), axis=-1)
    assert changed.any()
    n = min(tile_px - 1, max(1, round(tile_px * terrain_blend.EDGE_CONFORM_DEPTH)))
    _rows, cols = np.nonzero(changed[n:-n])
    assert ((cols + 0.5) / tile_px > 0.5).all(), "the bleed reached the side away from the +x neighbour"
    _rows, cols = np.nonzero(changed)
    assert ((cols + 0.5) / tile_px).mean() > 0.6
    # Cropped at THIS tile's offsets, so the bleed continues the neighbour's own tiles.
    expected = synthetic.textures[HIGH][oy : oy + tile_px, ox : ox + tile_px]
    zero = asset_source.get_blend_mask(*key[2][0][1:], tile_px) == 0
    assert zero.any() and np.array_equal(block[zero], expected[zero]), "alpha 0 must give the neighbour's own crop"


def test_a_constant_128_layer_mask_gives_the_exact_integer_midpoint(synthetic):
    size = asset_source.LOADED_TEXTURE_SIZE
    synthetic.masks["layer.png"] = np.full((size, size), 128, dtype=np.uint8)
    tile_px = 32
    ox, oy = _offsets(2, 2, tile_px)
    block = terrain_blend.blended_block((LOW, LAYER, ()), ox, oy, tile_px)
    base = _plain(LOW, 2, 2, tile_px).astype(np.int64)
    layer = synthetic.textures[LAYER][oy : oy + tile_px, ox : ox + tile_px].astype(np.int64)
    assert np.array_equal(block, ((base + layer + 1) >> 1).astype(np.uint8))


def test_an_empty_overlay_name_falls_back_to_the_neutral_mask(synthetic):
    assert terrain_blend.layer_mask_name(HIGH) == terrain_blend.LAYER_MASK_FALLBACK
    assert terrain_blend.layer_mask_name(HIGHER) == "higher.png"


def test_swapping_a_texture_reaches_the_block_through_clear_install_caches(synthetic):
    size = asset_source.LOADED_TEXTURE_SIZE
    synthetic.masks["layer.png"] = np.full((size, size), 128, dtype=np.uint8)
    key, tile_px = (LOW, LAYER, ()), 16
    ox, oy = _offsets(1, 1, tile_px)
    before = terrain_blend.blended_block(key, ox, oy, tile_px)
    synthetic.textures[LAYER] = 255 - synthetic.textures[LAYER]
    assert np.array_equal(terrain_blend.blended_block(key, ox, oy, tile_px), before), "expected the cached block"
    asset_source.clear_install_caches()
    assert not np.array_equal(terrain_blend.blended_block(key, ox, oy, tile_px), before)


def test_the_block_cache_stays_within_its_byte_budget_and_evicts_the_least_recent(synthetic, monkeypatch):
    monkeypatch.setattr(terrain_blend, "BLOCK_CACHE_BYTES", 3 * 32 * 32 * 3)
    key = (LOW, -1, ((HIGH, 2, 16),))
    first = terrain_blend.blended_block(key, *_offsets(0, 0, 32), 32)
    for x in range(1, 6):
        assert terrain_blend.blended_block(key, *_offsets(0, 0, 32), 32) is first, "a hit must refresh recency"
        terrain_blend.blended_block(key, *_offsets(x, 0, 32), 32)
    assert len(terrain_blend._blocks) == 3
    assert terrain_blend._block_bytes == 3 * 32 * 32 * 3
    assert (key, *_offsets(0, 0, 32), 32) in terrain_blend._blocks


def test_a_rebound_blend_table_is_never_served_stale_priorities(synthetic, monkeypatch):
    tids, layers = _grid(4, 4)
    tids[:, 2:] = HIGH
    keys = []
    for low_prio, high_prio in ((10, 20), (20, 10)):
        monkeypatch.setattr(
            terrain_blend, "BLEND_TABLE", {LOW: BlendEntry(low_prio, 0, ""), HIGH: BlendEntry(high_prio, 0, "")}
        )
        keys.append(influences(tids, layers, [1, 2], [1, 1]))
    assert keys[0][0] is not None and keys[0][1] is None
    assert keys[1][0] is None and keys[1][1] is not None


def test_influences_refuses_flat_mirrors_and_off_map_candidates(synthetic):
    tids, layers = _grid(4, 4)
    with pytest.raises(ValueError):
        influences(tids.reshape(-1), layers.reshape(-1), [1], [1])
    with pytest.raises(ValueError):
        influences(tids, layers, [-1], [0])


def test_clear_install_caches_drops_the_blend_loaders(tmp_path, monkeypatch):
    """Misses are cached too, so an empty fake install fills all three."""
    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(tmp_path))
    asset_source.clear_install_caches()
    loaders = (
        (asset_source.get_blendomatic, ()),
        (asset_source.get_blend_mask, (2, 0, 8)),
        (asset_source.get_terrain_mask_array, ("grass.png",)),
    )
    for loader, args in loaders:
        assert loader(*args) is None
        assert loader.cache_info().currsize == 1, loader.__name__
    asset_source.clear_install_caches()
    for loader, _args in loaders:
        assert loader.cache_info().currsize == 0, loader.__name__


def test_blocks_are_read_only(synthetic):
    block = terrain_blend.blended_block((LOW, -1, ((HIGH, 2, 0),)), 0, 0, 16)
    assert not block.flags.writeable


# --- KeyMap ---

KEY_TIDS = (LOW, HIGH, HIGHER, LAYER, 99)
KEY_LAYERS = (-1, -1, -1, LAYER, LOW, HIGH)


def _random_grid(rng, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    return rng.choice(KEY_TIDS, size=(h, w)).astype(np.int64), rng.choice(KEY_LAYERS, size=(h, w)).astype(np.int64)


def _derive(tids, layers):
    return lambda xs, ys: influences(tids, layers, xs, ys)


def _decode(key_map, ids) -> list:
    return terrain_blend.RectKeys(ids, key_map).as_list()


def _edit_round(rng, tids, layers) -> np.ndarray:
    """Rewrites about 1 tile in 8 of both grids in place (some writes keep the
    old value) and returns the flat indices whose tid or layer changed."""
    h, w = tids.shape
    idx = rng.choice(w * h, size=max(w * h // 8, 1), replace=False)
    ys, xs = np.divmod(idx, w)
    old = tids[ys, xs].copy(), layers[ys, xs].copy()
    tids[ys, xs] = rng.choice(KEY_TIDS, size=idx.size)
    layers[ys, xs] = rng.choice(KEY_LAYERS, size=idx.size)
    return idx[(tids[ys, xs] != old[0]) | (layers[ys, xs] != old[1])]


def _run_rounds(key_map, invalidate: bool, rounds: int = 12, seed: int = 20261012) -> list[bool]:
    """Per round, whether the map decoded to influences() of the live grids."""
    rng = np.random.default_rng(seed)
    w, h = key_map.w, key_map.h
    tids, layers = _random_grid(rng, w, h)
    key_map.bind((tids, layers))
    agreed = []
    for _ in range(rounds):
        idx = rng.integers(0, w * h, size=3 * w * h // 2)  # duplicates, and most of the map
        ys, xs = np.divmod(idx, w)
        got = _decode(key_map, key_map.lookup(idx, _derive(tids, layers)))
        agreed.append(got == influences(tids, layers, xs, ys))
        changed = _edit_round(rng, tids, layers)
        assert changed.size, "the round changed nothing (vacuous)"
        if invalidate:
            key_map.invalidate(changed)
    return agreed


def test_a_key_map_decodes_to_influences_across_edits(synthetic):
    """T1: lookup() through `keys` equals influences() on random grids and
    duplicated indices, over rounds of in-place edits each followed by
    invalidate() of the tiles that changed."""
    key_map = terrain_blend.KeyMap(11, 7)
    agreed = _run_rounds(key_map, invalidate=True)
    assert all(agreed), agreed
    assert len(key_map.keys) > 8, "too few distinct keys to mean anything"
    assert (key_map.ids == terrain_blend.NO_KEY).any() and (key_map.ids >= 0).any()
    for sid, key in enumerate(key_map.keys):
        assert key_map.own[sid] == key[0]
        assert key_map.extra[sid] == tuple(sorted({e[0] for e in key[2]} | ({key[1]} - {-1})))


def test_a_key_map_past_its_spec_cap_resets_and_still_decodes(synthetic, monkeypatch):
    """T1, second leg: with the cap at 4 specs the rounds cross the reset
    several times, and every lookup still decodes to influences()."""
    monkeypatch.setattr(terrain_blend, "KEY_MAP_MAX_SPECS", 4)
    key_map = terrain_blend.KeyMap(11, 7)
    resets = []
    real = key_map.reset
    monkeypatch.setattr(key_map, "reset", lambda: (resets.append(1), real())[1])
    assert all(_run_rounds(key_map, invalidate=True))
    assert len(resets) >= 3, resets


def test_a_key_map_without_invalidate_goes_stale(synthetic):
    """T2, the teeth of T1: the same rounds with no invalidate() decode to
    something influences() no longer says."""
    agreed = _run_rounds(terrain_blend.KeyMap(11, 7), invalidate=False)
    assert agreed[0] and not all(agreed), agreed


def test_binding_a_replaced_source_resets_and_the_same_sources_keep_every_id(synthetic, monkeypatch):
    """T3."""
    rng = np.random.default_rng(3)
    tids, layers = _random_grid(rng, 6, 5)
    derive = _derive(tids, layers)
    table = terrain_blend.BLEND_TABLE
    key_map = terrain_blend.KeyMap(6, 5)
    everything = np.arange(30)

    def fill(sources) -> np.ndarray:
        key_map.bind(sources)
        key_map.lookup(everything, derive)
        return key_map.ids

    ids = fill((tids, layers, table, derive))
    assert not (ids == terrain_blend.UNKNOWN).any()
    assert fill((tids, layers, table, derive)) is ids, "rebinding the same objects reset the map"
    swaps = (
        (tids.copy(), layers, table, derive),
        (tids, layers.copy(), table, derive),
        (tids, layers, dict(table), derive),
        (tids, layers, table, _derive(tids, layers)),
    )
    for i, sources in enumerate(swaps):
        key_map.bind(sources)
        assert (key_map.ids == terrain_blend.UNKNOWN).all() and not key_map.keys, f"source {i} replaced, map kept"
        fill(sources)
        assert key_map.ids is fill(sources)


@pytest.mark.parametrize("x", [0, 3, 6])
@pytest.mark.parametrize("y", [0, 2, 4])
def test_invalidate_marks_exactly_the_clipped_3x3(x, y):
    """T4: every corner, edge and the interior of a 7x5 map."""
    w, h = 7, 5
    key_map = terrain_blend.KeyMap(w, h)
    key_map.ids[:] = terrain_blend.NO_KEY
    key_map.invalidate([y * w + x])
    unknown = (key_map.ids == terrain_blend.UNKNOWN).reshape(h, w)
    want = np.zeros((h, w), dtype=bool)
    want[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = True
    assert np.array_equal(unknown, want)
    key_map.invalidate(np.zeros(0, dtype=np.int64))
    assert np.array_equal((key_map.ids == terrain_blend.UNKNOWN).reshape(h, w), want)
