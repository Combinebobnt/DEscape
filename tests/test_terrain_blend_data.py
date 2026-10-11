"""TASK-143 step 1: the committed blend table and the install loaders behind
terrain blending. The JSON checks need no install; the real-asset checks
skip without AOE2DE_INSTALL_PATH (conftest hides the configured install).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape import asset_source, iso_geometry, terrain_blend

MASK_COUNT = 26
TILE_PX_LADDER = (8, 16, 32, 64, 128)


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")


# --- committed JSON ---


def test_the_blend_table_covers_every_terrain_id():
    assert {t.value for t in TerrainId} <= set(terrain_blend.BLEND_TABLE)


def test_blend_priorities_are_unique_and_blend_types_index_the_mode_table():
    priorities = [e.priority for e in terrain_blend.BLEND_TABLE.values()]
    assert len(priorities) == len(set(priorities))
    assert {e.blend_type for e in terrain_blend.BLEND_TABLE.values()} <= set(range(len(terrain_blend.MODE_TABLE)))


def test_the_json_and_the_loaded_table_agree():
    raw = json.loads(terrain_blend.BLEND_MAP_PATH.read_text())["terrains"]
    assert len(raw) == len(terrain_blend.BLEND_TABLE) == 134
    assert raw["0"] == {"priority": 110, "blend_type": 0, "overlay_mask": "grass.png"}


# --- blendomatic parsing and resampling, no install ---


def _synthetic_blendomatic(n_modes: int = 2, n_tiles: int = 3) -> bytes:
    size = asset_source.BLENDOMATIC_TILE_SIZE
    out = bytearray(np.array([n_modes, n_tiles], dtype="<u4").tobytes())
    for mode in range(n_modes):
        out += np.array([size], dtype="<u4").tobytes() + bytes(n_tiles) + bytes(4 * size)
        for tile in range(n_tiles):
            out += bytes([(mode * 7 + tile) % 129]) * size
    return bytes(out)


def test_blendomatic_parses_a_synthetic_file_and_refuses_a_truncated_one():
    data = _synthetic_blendomatic()
    arr = asset_source._parse_blendomatic(data)
    assert arr.shape == (2, 3, asset_source.BLENDOMATIC_TILE_SIZE)
    assert int(arr[1, 2, 0]) == 9
    assert asset_source._parse_blendomatic(data[:-1]) is None
    assert asset_source._parse_blendomatic(data + b"\0") is None


def test_blendomatic_refuses_an_alpha_past_128():
    """The compose math (numpy and the kernel alike) holds only for 0..128."""
    data = bytearray(_synthetic_blendomatic())
    assert asset_source._parse_blendomatic(bytes(data)) is not None
    data[-1] = 129
    assert asset_source._parse_blendomatic(bytes(data)) is None


def _raster_uv() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(row, col, u, v) per blendomatic pixel, in file order."""
    rows_n, width = asset_source.BLENDOMATIC_ROWS, asset_source.BLENDOMATIC_WIDTH
    widths = [4 * min(r, rows_n - 1 - r) + 1 for r in range(rows_n)]
    rows = np.concatenate([np.full(w, r) for r, w in enumerate(widths)])
    cols = np.concatenate([np.arange(w) + (width - w) // 2 for w in widths])
    return rows, cols, (cols + 0.5) / width * 2 - 1, (rows + 0.5) / rows_n * 2 - 1


def test_the_raster_layout_is_a_full_97x49_diamond():
    rows, cols, _u, _v = _raster_uv()
    assert rows.size == asset_source.BLENDOMATIC_TILE_SIZE
    assert cols.min() == 0 and cols.max() == asset_source.BLENDOMATIC_WIDTH - 1


@pytest.mark.parametrize("tile_px", TILE_PX_LADDER)
def test_every_square_texel_takes_the_diamond_pixel_at_its_own_point(tile_px):
    """Every texel is defined and lands within one raster pixel of the exact
    diamond point square_to_diamond_local() gives it."""
    gather = asset_source.blendomatic_square_gather(tile_px)
    assert gather.shape == (tile_px, tile_px)
    assert gather.min() >= 0 and gather.max() < asset_source.BLENDOMATIC_TILE_SIZE
    rows, cols, _u, _v = _raster_uv()
    b, a = np.mgrid[0:tile_px, 0:tile_px]
    u, v = iso_geometry.square_to_diamond_local((a + 0.5) / tile_px, (b + 0.5) / tile_px)
    want_col = (u + 1) / 2 * asset_source.BLENDOMATIC_WIDTH - 0.5
    want_row = (v + 1) / 2 * asset_source.BLENDOMATIC_ROWS - 0.5
    # A row's span moves 2 columns per row, so clamping at a slanted edge can cost 2.
    assert np.abs(cols[gather] - want_col).max() <= 2.0
    assert np.abs(rows[gather] - want_row).max() <= 1.0


def test_square_to_diamond_local_inverts_the_diamond_sampling():
    tile_px = 64
    half_w, half_h = iso_geometry.half_dims(tile_px)
    dst_y, dst_x, src_y, src_x = iso_geometry.diamond_indices(tile_px)
    u, v = iso_geometry.square_to_diamond_local((src_x + 0.5) / tile_px, (src_y + 0.5) / tile_px)
    exact_u = (dst_x + 0.5 - half_w) / half_w
    exact_v = (dst_y + 0.5 - half_h) / half_h
    # A texel spans 1/tile_px in s and t, so 2/tile_px in u and v.
    assert np.abs(u - exact_u).max() <= 2 / tile_px
    assert np.abs(v - exact_v).max() <= 2 / tile_px


@pytest.mark.parametrize("tile_px", TILE_PX_LADDER)
def test_the_edge_fade_zeroes_each_bleeding_edge_and_spares_the_rest(tile_px):
    alpha = np.full((tile_px, tile_px), 128, dtype=np.uint8)
    n = max(1, round(tile_px * terrain_blend.EDGE_FADE_DEPTH))
    se = terrain_blend.fade_bleeding_edges(alpha, 1)
    assert (se[-1] == 0).all() and (se[: tile_px - n] == 128).all()
    assert (np.diff(se[:, 0].astype(int)) <= 0).all(), "the fade must fall monotonically toward the edge"
    both = terrain_blend.fade_bleeding_edges(alpha, 20)  # NE and SW
    assert (both[:, 0] == 0).all() and (both[:, -1] == 0).all() and (both[0, n:-n] == 128).all()
    for corner_only in range(16, 20):
        assert np.array_equal(terrain_blend.fade_bleeding_edges(alpha, corner_only), alpha)


def test_loaders_return_none_without_an_install(monkeypatch):
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()
    assert asset_source.get_blendomatic() is None
    assert asset_source.get_blend_mask(2, 0, 32) is None
    assert asset_source.get_terrain_mask_array("grass.png") is None


# --- real install ---


def test_real_blendomatic_parses_to_eof_as_9_modes_of_31_tiles():
    _require_install()
    tiles = asset_source.get_blendomatic()
    assert tiles is not None, "blendomatic_x1.dat missing or no longer parses to EOF"
    assert tiles.shape == (9, 31, asset_source.BLENDOMATIC_TILE_SIZE)
    assert int(tiles.max()) == 128
    for group in ((0, 1, 7, 8), (4, 6)):
        for mode in group[1:]:
            assert np.array_equal(tiles[group[0]], tiles[mode]), (group, mode)
    assert not np.array_equal(tiles[0], tiles[2])


@pytest.mark.parametrize("tile_px", TILE_PX_LADDER)
def test_real_blend_masks_resample_to_square_blocks(tile_px):
    _require_install()
    mask = asset_source.get_blend_mask(2, 30, tile_px)
    assert mask.shape == (tile_px, tile_px) and mask.dtype == np.uint8
    assert int(mask.max()) <= 128


_SINGLE_EDGE_DEPTH = {0: lambda s, t: 1 - t, 4: lambda s, t: 1 - s, 8: lambda s, t: s, 12: lambda s, t: t}
_SQUARE_EDGE = {0: np.s_[-1, :], 4: np.s_[:, -1], 8: np.s_[:, 0], 12: np.s_[0, :]}


def test_real_bleeding_edge_residue_is_the_assets_not_the_gathers():
    """TASK-283: on the raw 97x49 diamond, a single-edge tile's outermost
    raster row along its bleeding edge already keeps own weight (up to ~14 of
    128), and the square gather carries that over without adding any. So the
    seam residue is the asset's, which get_blend_mask's edge fade removes."""
    _require_install()
    tiles = asset_source.get_blendomatic()
    _rows, _cols, u, v = _raster_uv()
    s, t = (u - v + 1) / 2, (u + v + 1) / 2
    tile_px = 64
    gather = asset_source.blendomatic_square_gather(tile_px)
    raw_means = []
    for base, depth in _SINGLE_EDGE_DEPTH.items():
        band = depth(s, t) < 1 / asset_source.BLENDOMATIC_ROWS
        for tile in range(base, base + 4):
            for mode in (1, 2):
                raw = float(tiles[mode, tile][band].mean())
                square = float(tiles[mode, tile][gather][_SQUARE_EDGE[base]].mean())
                assert square <= raw + 1.0, (mode, tile, raw, square)
                raw_means.append(raw)
    assert max(raw_means) > 10, "the asset no longer keeps own weight on a bleeding edge"


@pytest.mark.parametrize("tile_px", TILE_PX_LADDER)
def test_real_masks_give_every_bleeding_edge_wholly_to_the_neighbour(tile_px):
    _require_install()
    gather = asset_source.blendomatic_square_gather(tile_px)
    n = max(1, round(tile_px * terrain_blend.EDGE_FADE_DEPTH))
    for tile in (*range(16), *range(20, 31)):
        edges = terrain_blend._BLEED_EDGES[tile]
        mask = asset_source.get_blend_mask(1, tile, tile_px)
        raw = asset_source.get_blendomatic()[1, tile][gather]
        far = {3: mask[-1, :], 1: mask[:, -1], 5: mask[:, 0], 7: mask[0, :]}
        for bit, edge in far.items():
            if edges >> bit & 1:
                assert not edge.any(), (tile, bit)
        inner = slice(n, tile_px - n)
        assert np.array_equal(mask[inner, inner], raw[inner, inner]), tile


def test_all_real_layer_masks_load_at_texture_size():
    _require_install()
    folder = asset_source.get_install_path() / asset_source.TERRAIN_MASK_SUBPATH
    names = sorted(p.name for p in folder.glob("*.png"))
    assert len(names) == MASK_COUNT
    modes = set()
    for name in names:
        from PIL import Image

        with Image.open(folder / name) as img:
            modes.add(img.mode)
        mask = asset_source.get_terrain_mask_array(name)
        assert mask is not None, name
        size = asset_source.LOADED_TEXTURE_SIZE
        assert mask.shape == (size, size) and mask.dtype == np.uint8, name
    assert {"L", "RGB"} <= modes, f"expected both L and RGB masks, got {modes}"


def test_every_named_layer_mask_and_the_fallback_exist():
    _require_install()
    names = {e.overlay_mask for e in terrain_blend.BLEND_TABLE.values() if e.overlay_mask}
    names.add(terrain_blend.LAYER_MASK_FALLBACK)
    missing = sorted(n for n in names if asset_source.get_terrain_mask_array(n) is None)
    assert not missing
