# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython port of iso_bench_numpy.py's per-tile isometric blit kernel.

Split the same way a real Phase 1 port would be: the per-tile Python-level
work (attribute access on the tile object, the lru_cache'd texture lookup)
stays in Python on both the numpy and Cython sides -- that part is
identical either way and isn't what a native backend buys you. What moves
into typed, nogil C loops here is the actual per-pixel numeric work: the
crop, the elevation-shade multiply+clip+cast, and the blit into the (much
larger) iso canvas -- the part that runs once per pixel per tile, tens of
thousands of times per full-map render.
"""

import numpy as np
cimport numpy as cnp

from descape import asset_source
from descape.terrain_palette import color_for_terrain_id
from iso_bench_numpy import iso_canvas_size_and_origin, tile_screen_origin

cnp.import_array()


cpdef void blit_tile_iso(
    unsigned char[:, :, ::1] img,
    const unsigned char[:, :, ::1] texture,
    int crop_x,
    int crop_y,
    int tile_px,
    int dest_x,
    int dest_y,
    double factor,
) noexcept nogil:
    """Crops a tile_px x tile_px block from `texture` at (crop_x, crop_y),
    multiplies by `factor` (elevation shading, matches render_tile()'s
    `1.0 + 0.08 * elevation`), clips to [0, 255], and blits into `img` at
    (dest_x, dest_y). Bounds-checks against img's own shape so an
    off-canvas tile (shouldn't happen given iso_canvas_size_and_origin, but
    this mirrors the numpy side's defensiveness) is silently skipped rather
    than corrupting memory -- required since this runs with the GIL
    released and boundscheck disabled."""
    cdef Py_ssize_t canvas_h = img.shape[0]
    cdef Py_ssize_t canvas_w = img.shape[1]
    if dest_y < 0 or dest_x < 0 or dest_y + tile_px > canvas_h or dest_x + tile_px > canvas_w:
        return

    cdef int row, col, ch
    # float32, not double: matches numpy's actual arithmetic precision on
    # the baseline side (block.astype(np.float32) * factor stays float32
    # under numpy>=2's NEP 50 scalar-promotion rules) -- computing this step
    # in double instead produces off-by-one rounding on ~0.02% of pixels
    # relative to the numpy kernel, confirmed by direct comparison during
    # this benchmark's development.
    cdef float val
    cdef float factor32 = <float>factor
    cdef unsigned char src

    for row in range(tile_px):
        for col in range(tile_px):
            for ch in range(3):
                src = texture[crop_y + row, crop_x + col, ch]
                val = <float>src * factor32
                if val < 0.0:
                    val = 0.0
                elif val > 255.0:
                    val = 255.0
                img[dest_y + row, dest_x + col, ch] = <unsigned char>val


def render_tile_iso_single(img_arr, tile, int tile_px, int elev_step, int origin_x, int origin_y):
    """One tile's worth of render_terrain_iso_cython's loop body, split out
    so tools/bench_iso_backend.py can time a single incremental composite
    into an already-allocated (warm) canvas -- the Phase 4
    bounded-region-redraw shape, not a full-map render. Mirrors
    iso_bench_numpy.render_tile_iso() exactly, one call per tile."""
    cdef unsigned char[:, :, ::1] img = img_arr
    cdef double factor = 1.0 + 0.08 * tile.elevation
    cdef int ox, oy, sx, sy
    cdef cnp.ndarray texture_arr

    texture = asset_source.get_terrain_texture_array(tile.terrain_id)
    if texture is not None:
        texture_arr = np.ascontiguousarray(texture)
        texture_size = texture_arr.shape[0]
        ox = (tile.x * tile_px) % texture_size
        oy = (tile.y * tile_px) % texture_size
    else:
        r, g, b = color_for_terrain_id(tile.terrain_id)
        texture_arr = np.full((tile_px, tile_px, 3), (r, g, b), dtype=np.uint8)
        ox, oy = 0, 0

    sx, sy = tile_screen_origin(tile.x, tile.y, tile.elevation, tile_px, elev_step, origin_x, origin_y)
    blit_tile_iso(img, texture_arr, ox, oy, tile_px, sx, sy, factor)


def render_terrain_iso_cython(tiles, int w, int h, int tile_px):
    """Full-map driver, the Cython-backend counterpart to
    iso_bench_numpy.render_terrain_iso_numpy. Same per-tile Python-level
    loop shape (texture lookup, crop-offset/screen-position math) as the
    numpy version -- only the inner blit_tile_iso call differs."""
    cdef int max_elev = 0
    tiles = list(tiles)
    for t in tiles:
        if t.elevation > max_elev:
            max_elev = t.elevation
    cdef int elev_step = tile_px // 4

    canvas_h, canvas_w, origin_x, origin_y = iso_canvas_size_and_origin(w, h, tile_px, max_elev, elev_step)
    img_arr = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for tile in tiles:
        render_tile_iso_single(img_arr, tile, tile_px, elev_step, origin_x, origin_y)

    return img_arr
