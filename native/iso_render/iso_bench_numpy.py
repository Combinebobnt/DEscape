"""Pure Python/numpy prototype of the per-tile isometric blit kernel -- the
Phase 0 baseline the Cython port in iso_kernel.pyx is benchmarked against.

Deliberately mirrors descape/render.py's render_tile() as closely as
possible (same crop_offset formula, same elevation-shading formula, same
block size) so the only thing this changes relative to the flat renderer is
*where* the shaded block gets blitted -- see this directory's README.md for
why diamond warping / skirt faces are out of scope here."""

from __future__ import annotations

import numpy as np

from descape import asset_source
from descape.render import _crop_offset
from descape.terrain_palette import color_for_terrain_id

# Matches the QTransform().scale(1, 0.5) baked into today's Flat isometric
# view (viewer.py's MapView.set_isometric) -- half-height footprint per tile.
ISO_HALF_W = 0.5  # fraction of tile_px
ISO_HALF_H = 0.25  # fraction of tile_px (0.5 squash * 0.5 half-width-again)


def iso_canvas_size_and_origin(w: int, h: int, tile_px: int, max_elev: int, elev_step: int):
    """Canvas (height, width) and the (origin_x, origin_y) offset needed so
    every tile's blit lands at a non-negative screen position, for the
    depth formula screen_x ∝ (x+y), screen_y ∝ (y-x) -- see the parent
    plan's decision #4 for why it's (y-x) and not the more usual (x+y)."""
    half_w = int(tile_px * ISO_HALF_W)
    half_h = int(tile_px * ISO_HALF_H)
    canvas_w = (w + h) * half_w + tile_px
    canvas_h = (w + h) * half_h + tile_px + max_elev * elev_step
    origin_x = 0  # (x+y) is already >= 0 for every tile
    origin_y = (w - 1) * half_h + max_elev * elev_step
    return canvas_h, canvas_w, origin_x, origin_y


def tile_screen_origin(x: int, y: int, elevation: int, tile_px: int, elev_step: int, origin_x: int, origin_y: int):
    half_w = int(tile_px * ISO_HALF_W)
    half_h = int(tile_px * ISO_HALF_H)
    sx = origin_x + (x + y) * half_w
    sy = origin_y + (y - x) * half_h - elevation * elev_step
    return sx, sy


def render_tile_iso(img: np.ndarray, tile, tile_px: int, elev_step: int, origin_x: int, origin_y: int) -> None:
    """The exact per-tile body of render.py's render_tile(), just blitted at
    an isometric screen position instead of tile.x*tile_px, tile.y*tile_px."""
    factor = 1.0 + 0.08 * tile.elevation

    texture = asset_source.get_terrain_texture_array(tile.terrain_id)
    if texture is not None:
        ox, oy = _crop_offset(tile.x, tile.y, texture.shape[0], tile_px)
        block = texture[oy : oy + tile_px, ox : ox + tile_px].astype(np.float32)
    else:
        r, g, b = color_for_terrain_id(tile.terrain_id)
        block = np.full((tile_px, tile_px, 3), (r, g, b), dtype=np.float32)

    shaded = np.clip(block * factor, 0, 255).astype(np.uint8)

    sx, sy = tile_screen_origin(tile.x, tile.y, tile.elevation, tile_px, elev_step, origin_x, origin_y)
    h, w = img.shape[:2]
    if sy < 0 or sx < 0 or sy + tile_px > h or sx + tile_px > w:
        return  # off-canvas -- shouldn't happen given iso_canvas_size_and_origin, but stay defensive
    img[sy : sy + tile_px, sx : sx + tile_px] = shaded


def render_terrain_iso_numpy(tiles, w: int, h: int, tile_px: int) -> np.ndarray:
    """Full-map driver -- the numpy-backend counterpart to Cython's
    render_terrain_iso_cython in iso_kernel.pyx. `tiles` is any iterable of
    objects with .x/.y/.elevation/.terrain_id (real MapManager.terrain tiles,
    or the synthetic stand-ins tools/bench_iso_backend.py builds for the
    480x480 worst case)."""
    max_elev = max((t.elevation for t in tiles), default=0)
    elev_step = tile_px // 4
    canvas_h, canvas_w, origin_x, origin_y = iso_canvas_size_and_origin(w, h, tile_px, max_elev, elev_step)
    img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for tile in tiles:
        render_tile_iso(img, tile, tile_px, elev_step, origin_x, origin_y)
    return img
