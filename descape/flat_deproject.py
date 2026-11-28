"""Top-down Flat's de-projection of an isometric frame onto the square grid
(TASK-350, the cliff follow-up to GH #151).

The inverse of the iso projection: an affine gather that sends a footprint's
screen diamond onto its Flat square, with the frame's hotspot on the
footprint centre. Tile offset (u, v) from that centre lands at screen
(hotspot_x + (u + v) * half_w, hotspot_y + (v - u) * half_w / 2), the same
axes the iso view draws (+x up-right, +y down-right). Ink outside the output
rect is clipped. unit_sprites._deprojected_icon() resizes the frame first so
the gather steps one source px per output px, which keeps it nearest and
leaves the filtering to the resize. Qt-free and image-library-free, like the
rest of the render path. Draw-only: no write, pick or footprint code reads it.
"""

from __future__ import annotations

import numpy as np


def deprojected(
    rgba: np.ndarray, hotspot_x: float, hotspot_y: float, half_w: float, tile_px: float, out_w: int, out_h: int,
    centre_dx: float = 0.0, centre_dy: float = 0.0,
) -> np.ndarray:
    """`rgba` (H, W, 4), whose tile diamond is 2 * half_w px wide, gathered
    nearest into an (out_h, out_w, 4) canvas of tile_px px per tile, the
    hotspot (continuous px, as a blit at anchor - hotspot places it) on the
    canvas centre moved by (centre_dx, centre_dy) output px: where the whole
    footprint's centre sits when the canvas is its map-clipped part. Samples
    off the source are transparent."""
    src_h, src_w = rgba.shape[:2]
    u = ((np.arange(out_w) + 0.5) - out_w / 2 - centre_dx) / tile_px
    v = ((np.arange(out_h) + 0.5) - out_h / 2 - centre_dy) / tile_px
    xs = np.floor(hotspot_x + (u[None, :] + v[:, None]) * half_w).astype(np.int64)
    ys = np.floor(hotspot_y + (v[:, None] - u[None, :]) * half_w / 2).astype(np.int64)
    inside = (xs >= 0) & (xs < src_w) & (ys >= 0) & (ys < src_h)
    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    out[inside] = rgba[ys[inside], xs[inside]]
    return out
