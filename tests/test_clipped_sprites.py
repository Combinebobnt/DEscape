"""Terrain must not cut sprite ink that overhangs nearer tiles (GH #107).

Stepped and Sloped used to paint terrain and objects interleaved per tile, so
every tile after a sprite's anchor repainted its ground over the overhang and
big doodads lost their lower ink along a tile-edge staircase. With sprites on
the composite is now two passes, all terrain then all objects, and a nearer
tile covers earlier ink only where it stands strictly higher than that ink's
base: a raised step still hides what is behind it.

Every test builds its expected image itself, from the terrain-only render and
its own alpha blend, and holds three renders to it on both composite
backends: the full render (render._paint_candidates_*, the reference loop)
and the chunk cache's stitched chunks, which on the native backend are the
kernel's own two passes. The sprite is a fake resolved through
unit_sprites.sprite_pieces_for, so it can carry what synthetic .sld bytes
cannot: a soft half-transparent band, a transparent margin and a hole.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from test_sprite_chunks import CHUNK, MAP_H, MAP_W, Unit, _scenario

from descape import asset_source, composite_backend, iso_geometry, render, render_cache, unit_sprites

from test_unit_sprites import CONST

STYLES = ("stepped", "sloped")
BACKENDS = ("numpy", "native")
MARK_CONST = CONST + 1  # the fake resolves no art for it, so it draws a mark
INK_H, INK_W = 192, 256
HOTSPOT = (INK_W // 2, INK_H // 2)
# The sprite unit's tile, mid-map: its ink reaches six depth rows past it.
UX = UY = 5


def _ink_rgba() -> np.ndarray:
    """Red-dominant per-pixel content (so a shifted blit shows, and no pixel
    can pass for terrain), a soft band along the bottom, a clear margin and a
    hole."""
    y, x = np.mgrid[0:INK_H, 0:INK_W]
    rgba = np.zeros((INK_H, INK_W, 4), dtype=np.uint8)
    rgba[..., 0] = 200 + (x * 3 + y) % 56
    rgba[..., 1] = (x * 5 + y * 11) % 256
    rgba[..., 2] = (x ^ y) % 256
    rgba[..., 3] = 255
    rgba[INK_H - 56 :, :, 3] = 96
    for edge in (np.s_[:8], np.s_[-8:]):
        rgba[edge, :, 3] = 0
        rgba[:, edge, 3] = 0
    rgba[60:80, 100:140, 3] = 0
    return rgba


INK = _ink_rgba()


@pytest.fixture(autouse=True)
def synthetic_textures(monkeypatch):
    """Every terrain id gets its own per-pixel texture and blending stays off,
    install or not: a blended edge keeps its neighbour's pixels under a
    recolour, so _own_pixels() would under-measure a tile's coverage."""
    cache: dict[int, np.ndarray] = {}

    def fake(terrain_id: int) -> np.ndarray:
        if terrain_id not in cache:
            y, x = np.mgrid[0:512, 0:512]
            t = terrain_id + 1
            cache[terrain_id] = np.stack(
                [(x * 7 + y * 13 + t * 31) % 256, (x * y + t * 17) % 256, (x ^ (y * t)) % 256], axis=-1
            ).astype(np.uint8)
        return cache[terrain_id]

    # set_install_path_override() calls this on the real lru_cache.
    fake.cache_clear = lambda: None
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", fake)
    no_blend = lambda: None  # noqa: E731
    no_blend.cache_clear = lambda: None
    monkeypatch.setattr(asset_source, "get_blendomatic", no_blend)


@pytest.fixture
def ink_sprite(monkeypatch):
    """CONST resolves to INK at every scale; nothing else resolves."""

    def pieces(unit_const, rotation, team_index, half_w, tree_scale=1.0, seed=None, hero_glow=False, art=None):
        if unit_const != CONST:
            return []
        return [unit_sprites.SpritePiece(unit_sprites.SpriteDraw(INK, *HOTSPOT), 0, 0)]

    monkeypatch.setattr(unit_sprites, "sprite_pieces_for", pieces)


@pytest.fixture(params=BACKENDS)
def backend(request):
    if request.param == "native" and not composite_backend.available():
        # Same contract as test_native_composite's native_kernel fixture.
        if os.environ.get("DESCAPE_REQUIRE_NATIVE") == "1":
            pytest.fail(f"DESCAPE_REQUIRE_NATIVE=1 but {composite_backend.unavailable_reason}")
        pytest.skip(composite_backend.unavailable_reason)
    with composite_backend.use_backend(request.param):
        yield request.param


def scenario_map(heights: dict | None = None, base: int = 0, marks=()):
    """The 12x12 synthetic map at `base`, `heights` overriding single tiles,
    with the sprite unit centred on (UX, UY) and a centred mark per `marks` tile."""
    units = [Unit(UX + 0.5, UY + 0.5, CONST), *(Unit(x + 0.5, y + 0.5, MARK_CONST) for x, y in marks)]
    scn = _scenario([[], units], elevation=base)
    for (x, y), level in (heights or {}).items():
        scn.map_manager.get_tile(x, y).elevation = level
    return scn


def _full(style: str, scn, **kwargs):
    """(image, elevations, corner_rise or None, proj) of the full render."""
    if style == "stepped":
        img, elevations, proj = render.render_terrain_iso_with_proj(scn, **kwargs)
        return img, elevations, None, proj
    return render.render_terrain_sloped_with_proj(scn, **kwargs)


def chunk_cache(style: str, scn):
    """A sprites-on chunk cache over scn, CHUNK px a chunk: the ink crosses
    several chunk seams."""
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    if style == "stepped":
        elevations, proj = render.elevations_and_proj(scn)
        return render_cache.IsoChunkCache(scn, elevations, proj, tile_px, chunk_px=CHUNK, sprites=True)
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return render_cache.SlopedChunkCache(scn, elevations, corner_rise, proj, tile_px, chunk_px=CHUNK, sprites=True)


def _stitched(style: str, scn) -> np.ndarray:
    """The whole canvas through the chunk cache: one composite per chunk, on
    the native backend one kernel call each."""
    cache = chunk_cache(style, scn)
    return cache.render_rect(0, 0, *cache.canvas_dims(0))


def _ink_origin(style: str, scn, elevations, corner_rise, proj) -> tuple[int, int]:
    """Top-left canvas pixel of the sprite, from the production anchor."""
    layer = render.sprite_draws_by_anchor(scn, proj, elevations, corner_rise=corner_rise)
    ((draw, ax, ay, _rise),) = layer.by_anchor[(UX, UY)]
    assert draw.rgba is INK
    return ax - draw.hotspot_x, ay - draw.hotspot_y


def _blend_ink(img: np.ndarray, x0: int, y0: int, keep: np.ndarray | None = None) -> np.ndarray:
    """This module's own alpha blend of INK onto a copy of img at (x0, y0).
    keep masks canvas pixels the ink must not touch."""
    out = img.copy()
    region = out[y0 : y0 + INK_H, x0 : x0 + INK_W].astype(np.uint16)
    assert region.shape[:2] == (INK_H, INK_W), "the sprite leaves the canvas -- move the unit"
    alpha = INK[..., 3:4].astype(np.uint16)
    blended = ((INK[..., :3].astype(np.uint16) * alpha + region * (255 - alpha)) // 255).astype(np.uint8)
    if keep is not None:
        covered = keep[y0 : y0 + INK_H, x0 : x0 + INK_W]
        blended[covered] = out[y0 : y0 + INK_H, x0 : x0 + INK_W][covered]
    out[y0 : y0 + INK_H, x0 : x0 + INK_W] = blended
    return out


def _ink_mask(shape, x0: int, y0: int, alpha_min: int, alpha_max: int) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=bool)
    mask[y0 : y0 + INK_H, x0 : x0 + INK_W] = (INK[..., 3] >= alpha_min) & (INK[..., 3] <= alpha_max)
    return mask


def _assert_renders(style: str, scn, full: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(full, expected):
        ys, xs = np.nonzero(np.any(full != expected, axis=-1))
        pytest.fail(f"{style} full render: {ys.size} px off the expected image, first at (y={ys[0]}, x={xs[0]})")
    stitched = _stitched(style, scn)
    h, w = stitched.shape[:2]
    if not np.array_equal(stitched, expected[:h, :w]):
        ys, xs = np.nonzero(np.any(stitched != expected[:h, :w], axis=-1))
        pytest.fail(f"{style} stitched chunks: {ys.size} px off the expected image, first at (y={ys[0]}, x={xs[0]})")


@pytest.mark.parametrize("style", STYLES)
def test_flat_ground_never_cuts_overhanging_ink(ink_sprite, backend, style):
    """The fix itself: on flat ground the whole sprite lies over the finished
    terrain, soft band included, however many later tiles it overhangs."""
    scn = scenario_map(base=2)
    terrain, *_ = _full(style, scn, with_units=False)
    full, elevations, corner_rise, proj = _full(style, scn, with_sprites=True)
    x0, y0 = _ink_origin(style, scn, elevations, corner_rise, proj)

    # Not vacuous: the ink reaches well below its own tile's diamond, onto later tiles.
    _tile_x, tile_y = iso_geometry.tile_screen_origin(UX, UY, 2, proj)
    assert y0 + INK_H > tile_y + 2 * proj.half_h + 3 * proj.half_h
    _assert_renders(style, scn, full, _blend_ink(terrain, x0, y0))


def _own_pixels(style: str, scn, tiles, terrain: np.ndarray) -> np.ndarray:
    """The canvas pixels the tiles' own terrain paints (each one's diamond or
    quad, and in Stepped its skirts): where the terrain-only render changes
    when only those tiles change terrain. Darken-only passes (seam, contact
    shadow) multiply whatever is under them, so they fall outside it unless
    they sit on the tiles' own pixels."""
    changed = [scn.map_manager.get_tile(x, y) for x, y in tiles]
    before = [tile.terrain_id for tile in changed]
    for tile in changed:
        tile.terrain_id = 6
    try:
        recoloured, *_ = _full(style, scn, with_units=False)
    finally:
        for tile, terrain_id in zip(changed, before, strict=True):
            tile.terrain_id = terrain_id
    return np.any(recoloured != terrain, axis=-1)


@pytest.mark.parametrize("style", STYLES)
def test_a_higher_nearer_tile_covers_ink_inside_its_own_coverage_only(ink_sprite, backend, style):
    """Two level-2 tiles in front of a level-0 sprite: inside those tiles' own
    painted pixels the image is the terrain-only render (opaque ink and soft
    band both gone, no ghost), everywhere else the ink is whole. Their
    contact shadows darken the ground behind them, never the sprite."""
    steps = ((UX - 1, UY + 1), (UX - 2, UY + 2))
    scn = scenario_map(dict.fromkeys(steps, 2))
    terrain, *_ = _full(style, scn, with_units=False)
    full, elevations, corner_rise, proj = _full(style, scn, with_sprites=True)
    x0, y0 = _ink_origin(style, scn, elevations, corner_rise, proj)
    coverage = _own_pixels(style, scn, steps, terrain)

    opaque = _ink_mask(full.shape, x0, y0, 255, 255)
    soft = _ink_mask(full.shape, x0, y0, 1, 254)
    assert (coverage & opaque).any() and (coverage & soft).any(), "the step covers no ink -- vacuous"
    assert (opaque & ~coverage).any() and (soft & ~coverage).any(), "the step covers all the ink -- vacuous"
    _assert_renders(style, scn, full, _blend_ink(terrain, x0, y0, keep=coverage))


@pytest.mark.parametrize("style", STYLES)
def test_ink_based_above_nearer_ground_survives(ink_sprite, backend, style):
    """A sprite on a level-2 plateau tile over level-0 ground: nothing in
    front of it stands higher than its base, so no ink is covered. The far
    level-3 tile only makes the occlusion pass run at all."""
    scn = scenario_map({(UX, UY): 2, (MAP_W - 1, 0): 3})
    terrain, *_ = _full(style, scn, with_units=False)
    full, elevations, corner_rise, proj = _full(style, scn, with_sprites=True)
    x0, y0 = _ink_origin(style, scn, elevations, corner_rise, proj)
    _assert_renders(style, scn, full, _blend_ink(terrain, x0, y0))


def _tile_diamond(shape, tx: int, ty: int, level: int, proj) -> np.ndarray:
    dst_y, dst_x, _sy, _sx = iso_geometry.diamond_indices(render.tile_pixels_for_map(MAP_W, MAP_H))
    base_x, base_y = iso_geometry.tile_screen_origin(tx, ty, level, proj)
    mask = np.zeros(shape[:2], dtype=bool)
    mask[base_y + dst_y, base_x + dst_x] = True
    return mask


@pytest.mark.parametrize("style", STYLES)
def test_marks_keep_their_depth_order_against_a_sprite(ink_sprite, backend, style):
    """Objects still paint in depth order among themselves: a mark on a nearer
    tile paints over the sprite's ink, a mark on a farther tile under it."""
    near, far = (UX - 1, UY + 1), (UX + 1, UY - 1)
    scn = scenario_map(base=1, marks=(near, far))
    terrain, *_ = _full(style, scn, with_units=False)
    full, elevations, corner_rise, proj = _full(style, scn, with_sprites=True)
    x0, y0 = _ink_origin(style, scn, elevations, corner_rise, proj)
    colour = render.unit_mark_color(scn.unit_manager.units[1][1], scn.player_colors[1])

    near_px, far_px = (_tile_diamond(full.shape, *tile, 1, proj) for tile in (near, far))
    opaque = _ink_mask(full.shape, x0, y0, 255, 255)
    assert (near_px & opaque).any() and (far_px & opaque).any(), "a mark misses the ink -- vacuous"
    expected = terrain.copy()
    expected[far_px] = colour
    expected = _blend_ink(expected, x0, y0)
    expected[near_px] = colour
    _assert_renders(style, scn, full, expected)
