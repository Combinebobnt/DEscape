"""Iso texture continuity (TASK-201): a same-terrain area in Stepped and
Sloped reads as one unrolled texture, with no per-tile seam.

The oracle is a synthetic COORDINATE texture: every texel stores its own
(column, row), so a rendered canvas pixel decodes straight back to the texel
it sampled. Tile seams then show up as a texel jump between two screen
neighbours that sit in different tiles. Before the fix
iso_geometry._inverse_sample read only the centre quarter of each crop with
its axes turned 45 degrees to _crop_offset's progression, and every seam
jumped ~71 texels at tile_px=64 while within-tile steps stayed under 1.5.

Everything here goes through the real chunk caches, so DESCAPE_COMPOSITE
picks which backend is under test (run the file under both).
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import asset_source, iso_geometry, native_composite, render
from descape.render import tile_pixels_for_map
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache
from descape.view_layers import LayerState
from testkit.fakes import FakeScenario, SyntheticTile

TEXTURE_SIZE = 512
MAP_W = MAP_H = 8
TERRAIN_ID = 1
# Marker bits in the blue channel's high bits: a pixel whose blue does not
# carry them was painted by something other than a raw texel (background,
# darkening, a grid line) and is left out of every measurement.
MARKER = 0xF0
# The new basis' within-tile step: one pixel down is (-2, +2) texels.
WITHIN_TILE_MAX = float(np.hypot(2, 2))
# Sloped's per-column resample is a linearization (iso_geometry
# sloped_quad_indices), so a stretched or squashed column can step a texel
# further than the flat case on either side of a seam.
SLOPED_MARGIN = 1.5


def _coord_texture() -> np.ndarray:
    y, x = np.mgrid[0:TEXTURE_SIZE, 0:TEXTURE_SIZE]
    return np.stack([x & 255, y & 255, MARKER | (x >> 8) | ((y >> 8) << 1)], axis=-1).astype(np.uint8)


@pytest.fixture
def coord_texture(monkeypatch):
    texture = _coord_texture()
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", lambda _tid: texture)
    monkeypatch.setattr(asset_source, "get_terrain_average_color", lambda _tid: None)
    return texture


def _scenario(elevation=lambda x, y: 0) -> FakeScenario:
    tiles = [
        SyntheticTile(x=x, y=y, elevation=elevation(x, y), terrain_id=TERRAIN_ID)
        for y in range(MAP_H)
        for x in range(MAP_W)
    ]
    return FakeScenario(MAP_W, MAP_H, tiles, [[] for _ in range(9)])


def _sloping(x: int, y: int) -> int:
    # Neighbours differ by at most 1, with a twist term, so tiles really slope.
    return x // 2 + y // 3


def _iso_cache(scn, **kwargs) -> IsoChunkCache:
    elevations, proj = render.elevations_and_proj(scn)
    return IsoChunkCache(scn, elevations, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _sloped_cache(scn, **kwargs) -> SlopedChunkCache:
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return SlopedChunkCache(
        scn, elevations, corner_rise, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs
    )


def _flat_cache(scn, **kwargs) -> FlatChunkCache:
    return FlatChunkCache(scn, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _level_canvas(cache, mip: int = 0) -> np.ndarray:
    w, h = cache.canvas_dims(mip)
    return cache.render_rect(0, 0, w, h, mip=mip)


def _decode(img: np.ndarray):
    r, g, b = (img[..., i].astype(np.int64) for i in range(3))
    valid = (b & 0xFC) == MARKER
    col = r | ((b & 1) << 8)
    row = g | (((b >> 1) & 1) << 8)
    return col, row, valid


def _wrap(d: np.ndarray) -> np.ndarray:
    # Toroidal difference: the texture tiles, so a crop wrapping from the
    # last block to the first is a step of 1, not TEXTURE_SIZE - 1.
    return (d + TEXTURE_SIZE // 2) % TEXTURE_SIZE - TEXTURE_SIZE // 2


def _steps(img: np.ndarray, tile_px: int) -> tuple[float, float, int]:
    """(worst within-tile step, worst across-seam step, seam pair count) over
    every 4-neighbour pair of texel pixels, in texels. A pixel's tile is
    the crop block its texel lies in: adjacent tiles never share one."""
    col, row, valid = _decode(img)
    block = (col // tile_px) * (TEXTURE_SIZE // tile_px) + row // tile_px
    within, seam, n_seam = 0.0, 0.0, 0
    for a, b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        both = valid[a] & valid[b]
        dist = np.hypot(_wrap(col[a] - col[b]), _wrap(row[a] - row[b]))[both]
        crosses = (block[a] != block[b])[both]
        if (~crosses).any():
            within = max(within, float(dist[~crosses].max()))
        if crosses.any():
            seam = max(seam, float(dist[crosses].max()))
            n_seam += int(crosses.sum())
    return within, seam, n_seam


@pytest.fixture
def unshaded(monkeypatch):
    """Slope shading scales the colour, which would break the coordinate
    decode. A factor of exactly 1 leaves every texel byte untouched on both
    backends: native_composite fetches the factor through render too."""
    monkeypatch.setattr(
        render, "_slope_shade",
        lambda tile_px, *_a: np.ones(iso_geometry.diamond_indices(tile_px)[0].size, dtype=np.float32),
    )


def test_a_flat_stepped_map_has_no_tile_seam(coord_texture) -> None:
    cache = _iso_cache(_scenario())
    for mip in cache.mip_levels():
        tile_px = cache.mip_tile_px(mip)
        img = _level_canvas(cache, mip)
        # Nothing but raw texels may be on the map: every diamond pixel decodes.
        n_diamond = iso_geometry.diamond_indices(tile_px)[0].size
        assert int(_decode(img)[2].sum()) == MAP_W * MAP_H * n_diamond, f"mip {mip}: a pixel did not decode"
        within, seam, n_seam = _steps(img, tile_px)
        assert n_seam > MAP_W * MAP_H, f"mip {mip}: too few seam pairs ({n_seam}) to measure"
        assert within <= WITHIN_TILE_MAX + 1e-9, f"mip {mip} (tile_px {tile_px}): within-tile step {within}"
        assert seam <= within, f"mip {mip} (tile_px {tile_px}): seam step {seam} > within-tile {within}"


def test_a_sloped_map_has_no_tile_seam(coord_texture, unshaded) -> None:
    scn = _scenario(_sloping)
    _elevations, corner_rise, _proj = render.sloped_elevations_and_proj(scn)
    assert len(np.unique(corner_rise)) > 2, "fixture no longer slopes"
    cache = _sloped_cache(scn)
    for mip in cache.mip_levels():
        tile_px = cache.mip_tile_px(mip)
        img = _level_canvas(cache, mip)
        within, seam, n_seam = _steps(img, tile_px)
        assert n_seam > MAP_W * MAP_H, f"mip {mip}: too few seam pairs ({n_seam}) to measure"
        assert seam <= within + SLOPED_MARGIN, (
            f"mip {mip} (tile_px {tile_px}): seam step {seam} > within-tile {within} + {SLOPED_MARGIN}"
        )


@pytest.mark.parametrize("tile_px", [16, 32, 64, 128])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("drop_px", [1, 5, 16])
def test_a_skirt_top_row_is_its_diamond_edge_texel(tile_px, side, drop_px) -> None:
    """Every skirt column repeats one texel straight down, and that texel is
    the one the diamond shows directly above it, so the face continues the
    top without a seam."""
    dst_y, dst_x, src_y, src_x = iso_geometry.skirt_quad_indices(tile_px, drop_px, side)
    d_dst_y, d_dst_x, d_src_y, d_src_x = iso_geometry.diamond_indices(tile_px)
    texel = {(int(y), int(x)): (int(sy), int(sx)) for y, x, sy, sx in zip(d_dst_y, d_dst_x, d_src_y, d_src_x, strict=True)}
    _tops, bottoms, _used = iso_geometry._diamond_column_edges(tile_px)
    for y, x, sy, sx in zip(dst_y, dst_x, src_y, src_x, strict=True):
        edge = (int(bottoms[x]), int(x))
        assert int(y) > edge[0]
        assert (int(sy), int(sx)) == texel[edge], f"skirt pixel ({int(y)}, {int(x)}) is not its edge texel"


def _legacy_inverse_sample(dst_x, dst_y, half_w: int, half_h: int, tile_px: int):
    """The pre-TASK-201 centre-quarter basis, verbatim, as the "before"."""
    u = (dst_x + 0.5 - half_w) / half_w
    v = (dst_y + 0.5 - half_h) / half_h
    a = (u + v) / 2.0
    b = (v - u) / 2.0
    src_x = np.clip(np.floor((a + 1.0) / 2.0 * tile_px), 0, tile_px - 1).astype(np.int64)
    src_y = np.clip(np.floor((b + 1.0) / 2.0 * tile_px), 0, tile_px - 1).astype(np.int64)
    return src_y, src_x


_SKIRT_QUAD_INDICES = iso_geometry.skirt_quad_indices


def _legacy_skirt_quad_indices(tile_px: int, drop_px: int, side: str):
    """The pre-TASK-201 skirt: same dst, but each column sampled one row
    below the diamond's edge."""
    dst_y, dst_x, _src_y, _src_x = _SKIRT_QUAD_INDICES(tile_px, drop_px, side)
    half_w, half_h = iso_geometry.half_dims(tile_px)
    _tops, bottoms, _used = iso_geometry._diamond_column_edges(tile_px)
    src_y, src_x = _legacy_inverse_sample(dst_x, bottoms[dst_x] + 1, half_w, half_h, tile_px)
    return dst_y, dst_x, src_y, src_x


def _clear_geometry_caches() -> None:
    for fn in (
        iso_geometry.diamond_indices,
        _SKIRT_QUAD_INDICES,
        iso_geometry.sloped_quad_indices,
        native_composite.stepped_tables,
        native_composite.sloped_shapes,
    ):
        fn.cache_clear()


def _renders(builder, scn, texture) -> dict[str, np.ndarray]:
    asset_source.get_terrain_texture_array = lambda _tid: texture
    out = {
        "on": _level_canvas(builder(scn)).copy(),
        "off": _level_canvas(builder(scn, layers=LayerState(terrain_textures=False))).copy(),
    }
    asset_source.get_terrain_texture_array = lambda _tid: None
    out["no_install"] = _level_canvas(builder(scn)).copy()
    return out


@pytest.mark.parametrize("builder", [_flat_cache, _iso_cache, _sloped_cache])
def test_only_texture_sampling_moved(builder, coord_texture) -> None:
    """Textures off and no install are byte-identical before vs after the
    basis change, in all three styles: the fix moved texture sampling and
    nothing else. Slope shading's tile_uv_fractions keeps its own basis.
    "Before" is the legacy basis and skirt row swapped back in for one
    render; textures on must differ under it in Stepped and Sloped, or the
    swap did nothing."""
    # coord_texture's monkeypatch restores the real getter; _renders rebinds it.
    scn = _scenario(lambda x, y: (x + y) % 3)
    after = _renders(builder, scn, coord_texture)
    current = iso_geometry._inverse_sample
    iso_geometry._inverse_sample = _legacy_inverse_sample
    iso_geometry.skirt_quad_indices = _legacy_skirt_quad_indices
    _clear_geometry_caches()
    try:
        before = _renders(builder, scn, coord_texture)
    finally:
        iso_geometry._inverse_sample = current
        iso_geometry.skirt_quad_indices = _SKIRT_QUAD_INDICES
        _clear_geometry_caches()
    assert np.array_equal(before["off"], after["off"])
    assert np.array_equal(before["no_install"], after["no_install"])
    assert np.array_equal(after["off"], after["no_install"])
    if builder is _flat_cache:
        assert np.array_equal(before["on"], after["on"]), "Flat blits whole crops and must not move"
    else:
        assert not np.array_equal(before["on"], after["on"]), "the legacy swap did not reach the render"
