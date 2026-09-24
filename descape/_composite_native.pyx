# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Native composite kernel. Built by tools/build_native.py; selected by
descape/composite_backend.py, which falls back to numpy when this is missing
or its KERNEL_ABI doesn't match. Every function must stay byte-identical to
the numpy code it replaces (tests/test_native_composite.py)."""

from libc.stdint cimport int64_t, uint8_t

# Bump on any signature or semantics change, together with
# composite_backend.EXPECTED_KERNEL_ABI, so a stale build is never used.
KERNEL_ABI = 1


def paint_diamond(uint8_t[:, :, :] img, Py_ssize_t base_y, Py_ssize_t base_x,
                  const int64_t[::1] dst_y, const int64_t[::1] dst_x,
                  const int64_t[::1] src_y, const int64_t[::1] src_x,
                  const uint8_t[:, :, :] top_block):
    """render._clipped_paint(img, base_y, base_x, dst_y, dst_x,
    top_block[src_y, src_x]): an opaque uint8 scatter, dropping any
    destination pixel outside img."""
    cdef Py_ssize_t k, n = dst_y.shape[0], h = img.shape[0], w = img.shape[1]
    cdef Py_ssize_t th = top_block.shape[0], tw = top_block.shape[1]
    cdef Py_ssize_t py, px, sy, sx
    cdef bint bad_src = False
    if dst_x.shape[0] != n or src_y.shape[0] != n or src_x.shape[0] != n:
        raise ValueError("index arrays differ in length")
    if img.shape[2] != 3 or top_block.shape[2] != 3:
        raise ValueError("img and top_block must have 3 channels")
    with nogil:
        for k in range(n):
            py = base_y + dst_y[k]
            px = base_x + dst_x[k]
            if py < 0 or py >= h or px < 0 or px >= w:
                continue
            sy = src_y[k]
            sx = src_x[k]
            # numpy's gather raises here; boundscheck=False would read junk.
            if sy < 0 or sy >= th or sx < 0 or sx >= tw:
                bad_src = True
                break
            img[py, px, 0] = top_block[sy, sx, 0]
            img[py, px, 1] = top_block[sy, sx, 1]
            img[py, px, 2] = top_block[sy, sx, 2]
    if bad_src:
        raise IndexError("source index outside top_block")
