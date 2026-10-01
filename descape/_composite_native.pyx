# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Native composite kernel. Built by tools/build_native.py; selected by
descape/composite_backend.py, which falls back to numpy when this is missing
or its KERNEL_ABI doesn't match. Every function must stay byte-identical to
the numpy code it replaces (tests/test_native_composite.py, and
tests/test_sld_native.py for the SLD walk and decode at the end).

Arithmetic mirrors numpy's dtypes exactly: opaque scatters are uint8, darken
and slope shade are float32 multiplies truncated to uint8 (never a double
temporary), and the lerp and rgba blit are integer (t*a + d*(255-a)) // 255.
No index producer repeats a destination within one call, so writing in place
reads each pixel once, the same as numpy's fancy-index read-modify-write."""

from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, uintptr_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset

# Bump on any signature or semantics change, together with
# composite_backend.EXPECTED_KERNEL_ABI, so a stale build is never used.
KERNEL_ABI = 4


cdef inline bint _gather_scatter(
    uint8_t[:, :, :] img, Py_ssize_t base_y, Py_ssize_t base_x,
    const int64_t[::1] dst_y, const int64_t[::1] dst_x,
    const int64_t[::1] src_y, const int64_t[::1] src_x,
    const uint8_t[:, :, :] block, const uint8_t[::1] lut, bint use_lut,
) noexcept nogil:
    """img[base+dst] = (lut of) block[src], clipped. False on a source index
    outside block, which numpy's gather would raise on."""
    cdef Py_ssize_t k, n = dst_y.shape[0], h = img.shape[0], w = img.shape[1]
    cdef Py_ssize_t th = block.shape[0], tw = block.shape[1]
    cdef Py_ssize_t py, px, sy, sx
    for k in range(n):
        py = base_y + dst_y[k]
        px = base_x + dst_x[k]
        if py < 0 or py >= h or px < 0 or px >= w:
            continue
        sy = src_y[k]
        sx = src_x[k]
        if sy < 0 or sy >= th or sx < 0 or sx >= tw:
            return False
        if use_lut:
            img[py, px, 0] = lut[block[sy, sx, 0]]
            img[py, px, 1] = lut[block[sy, sx, 1]]
            img[py, px, 2] = lut[block[sy, sx, 2]]
        else:
            img[py, px, 0] = block[sy, sx, 0]
            img[py, px, 1] = block[sy, sx, 1]
            img[py, px, 2] = block[sy, sx, 2]
    return True


cdef inline void _darken(
    uint8_t[:, :, :] img, Py_ssize_t base_y, Py_ssize_t base_x,
    const int64_t[::1] dst_y, const int64_t[::1] dst_x, const float[::1] factors,
) noexcept nogil:
    """render._clipped_darken: uint8 * float32, truncated back to uint8."""
    cdef Py_ssize_t k, n = dst_y.shape[0], h = img.shape[0], w = img.shape[1]
    cdef Py_ssize_t py, px
    cdef float f
    for k in range(n):
        py = base_y + dst_y[k]
        px = base_x + dst_x[k]
        if py < 0 or py >= h or px < 0 or px >= w:
            continue
        f = factors[k]
        img[py, px, 0] = <uint8_t>(<float>img[py, px, 0] * f)
        img[py, px, 1] = <uint8_t>(<float>img[py, px, 1] * f)
        img[py, px, 2] = <uint8_t>(<float>img[py, px, 2] * f)


def _check_lengths(Py_ssize_t n, *others) -> None:
    for m in others:
        if m != n:
            raise ValueError("index arrays differ in length")


def _check_rgb(img, block) -> None:
    if img.shape[2] != 3 or block.shape[2] != 3:
        raise ValueError("img and block must have 3 channels")


def paint_diamond(uint8_t[:, :, :] img, Py_ssize_t base_y, Py_ssize_t base_x,
                  const int64_t[::1] dst_y, const int64_t[::1] dst_x,
                  const int64_t[::1] src_y, const int64_t[::1] src_x,
                  const uint8_t[:, :, :] top_block):
    """render._clipped_paint(img, base_y, base_x, dst_y, dst_x,
    top_block[src_y, src_x]): an opaque uint8 scatter, dropping any
    destination pixel outside img."""
    cdef bint ok
    _check_lengths(dst_y.shape[0], dst_x.shape[0], src_y.shape[0], src_x.shape[0])
    _check_rgb(img, top_block)
    with nogil:
        ok = _gather_scatter(img, base_y, base_x, dst_y, dst_x, src_y, src_x, top_block, None, False)
    if not ok:
        raise IndexError("source index outside top_block")


def render_tile_iso(uint8_t[:, :, :] img, Py_ssize_t base_y, Py_ssize_t base_x,
                    const uint8_t[:, :, :] top_block, list skirts, tuple diamond, list darkens):
    """One Stepped tile, every paint in render._render_tile_iso's order:
    skirts (dst_y, dst_x, src_y, src_x, lut), then the diamond (dst_y,
    dst_x, src_y, src_x), then each darken (dst_y, dst_x, factors, ...) in
    list order (seams, seam apex, shadow bands, apex wedge, tips). The caller
    owns every gate; this only paints what it is handed."""
    cdef const int64_t[::1] dy
    cdef const int64_t[::1] dx
    cdef const int64_t[::1] sy
    cdef const int64_t[::1] sx
    cdef const uint8_t[::1] lut
    cdef const float[::1] factors
    cdef bint ok
    _check_rgb(img, top_block)
    for op in skirts:
        dy, dx, sy, sx, lut = op[0], op[1], op[2], op[3], op[4]
        _check_lengths(dy.shape[0], dx.shape[0], sy.shape[0], sx.shape[0])
        if lut.shape[0] != 256:
            raise ValueError("skirt lut must have 256 entries")
        with nogil:
            ok = _gather_scatter(img, base_y, base_x, dy, dx, sy, sx, top_block, lut, True)
        if not ok:
            raise IndexError("source index outside top_block")
    dy, dx, sy, sx = diamond[0], diamond[1], diamond[2], diamond[3]
    _check_lengths(dy.shape[0], dx.shape[0], sy.shape[0], sx.shape[0])
    with nogil:
        ok = _gather_scatter(img, base_y, base_x, dy, dx, sy, sx, top_block, None, False)
    if not ok:
        raise IndexError("source index outside top_block")
    for op in darkens:
        dy, dx, factors = op[0], op[1], op[2]
        _check_lengths(dy.shape[0], dx.shape[0], factors.shape[0])
        with nogil:
            _darken(img, base_y, base_x, dy, dx, factors)


def paint_sloped(uint8_t[:, :, :] img, Py_ssize_t base_y, Py_ssize_t base_x,
                 const int64_t[::1] dst_y, const int64_t[::1] dst_x,
                 const int64_t[::1] src_y, const int64_t[::1] src_x, const int64_t[::1] uv_idx,
                 const uint8_t[:, :, :] top_block, const float[::1] shade):
    """render._render_tile_sloped's paint: top_block[src] * shade[uv_idx] in
    float32, clipped to [0, 255], truncated to uint8, scattered with a clip."""
    cdef Py_ssize_t k, n = dst_y.shape[0], h = img.shape[0], w = img.shape[1]
    cdef Py_ssize_t th = top_block.shape[0], tw = top_block.shape[1], ns = shade.shape[0]
    cdef Py_ssize_t py, px, sy, sx, u, c
    cdef float s, v
    cdef bint bad = False
    _check_lengths(n, dst_x.shape[0], src_y.shape[0], src_x.shape[0], uv_idx.shape[0])
    _check_rgb(img, top_block)
    with nogil:
        for k in range(n):
            py = base_y + dst_y[k]
            px = base_x + dst_x[k]
            if py < 0 or py >= h or px < 0 or px >= w:
                continue
            sy = src_y[k]
            sx = src_x[k]
            u = uv_idx[k]
            if sy < 0 or sy >= th or sx < 0 or sx >= tw or u < 0 or u >= ns:
                bad = True
                break
            s = shade[u]
            for c in range(3):
                v = <float>top_block[sy, sx, c] * s
                if v < 0:
                    v = 0
                elif v > 255:
                    v = 255
                img[py, px, c] = <uint8_t>v
    if bad:
        raise IndexError("source or uv index outside top_block/shade")


def paint_solid(uint8_t[:, :, :] img, Py_ssize_t base_y, Py_ssize_t base_x,
                const int64_t[::1] dst_y, const int64_t[::1] dst_x, uint8_t r, uint8_t g, uint8_t b):
    """render._clipped_paint with one colour for every pixel (farm outlines,
    unit marks)."""
    cdef Py_ssize_t k, n = dst_y.shape[0], h = img.shape[0], w = img.shape[1]
    cdef Py_ssize_t py, px
    _check_lengths(n, dst_x.shape[0])
    if img.shape[2] != 3:
        raise ValueError("img must have 3 channels")
    with nogil:
        for k in range(n):
            py = base_y + dst_y[k]
            px = base_x + dst_x[k]
            if py < 0 or py >= h or px < 0 or px >= w:
                continue
            img[py, px, 0] = r
            img[py, px, 1] = g
            img[py, px, 2] = b


def lerp(uint8_t[:, :, :] img, Py_ssize_t base_y, Py_ssize_t base_x,
         const int64_t[::1] dst_y, const int64_t[::1] dst_x,
         const uint16_t[:, :] target, const uint16_t[:, :] alpha):
    """render._clipped_lerp with per-pixel (n, 3) target and (n, 1) alpha:
    (t*a + d*(255-a)) // 255 in integers, clipped."""
    cdef Py_ssize_t k, n = dst_y.shape[0], h = img.shape[0], w = img.shape[1]
    cdef Py_ssize_t py, px, c
    cdef unsigned int a
    _check_lengths(n, dst_x.shape[0], target.shape[0], alpha.shape[0])
    if img.shape[2] != 3 or target.shape[1] != 3 or alpha.shape[1] != 1:
        raise ValueError("img and target must have 3 channels, alpha 1")
    with nogil:
        for k in range(n):
            py = base_y + dst_y[k]
            px = base_x + dst_x[k]
            if py < 0 or py >= h or px < 0 or px >= w:
                continue
            a = alpha[k, 0]
            for c in range(3):
                img[py, px, c] = <uint8_t>((target[k, c] * a + img[py, px, c] * (255 - a)) // 255)


def blit_rgba(uint8_t[:, :, :] img, Py_ssize_t base_y, Py_ssize_t base_x, const uint8_t[:, :, :] rgba):
    """render._clipped_paint_rgba: integer alpha blend of an (h, w, 4) block
    at (base_y, base_x), clipped to img by rectangle intersection."""
    cdef Py_ssize_t h = rgba.shape[0], w = rgba.shape[1], ih = img.shape[0], iw = img.shape[1]
    cdef Py_ssize_t sy0, sx0, sy1, sx1, y, x, c, py, px
    cdef unsigned int a
    if img.shape[2] != 3 or rgba.shape[2] != 4:
        raise ValueError("img must have 3 channels and rgba 4")
    sy0 = -base_y if base_y < 0 else 0
    sx0 = -base_x if base_x < 0 else 0
    sy1 = h if h < ih - base_y else ih - base_y
    sx1 = w if w < iw - base_x else iw - base_x
    if sy0 >= sy1 or sx0 >= sx1:
        return
    with nogil:
        for y in range(sy0, sy1):
            py = base_y + y
            for x in range(sx0, sx1):
                px = base_x + x
                a = rgba[y, x, 3]
                for c in range(3):
                    img[py, px, c] = <uint8_t>((rgba[y, x, c] * a + img[py, px, c] * (255 - a)) // 255)


# ---------------------------------------------------------------------------
# N2: one call per rect. descape/native_composite.py builds every input; the
# per-tile gates, paint order and arithmetic below mirror
# render._paint_tile_and_units_iso/_sloped exactly.

ctypedef struct Canvas:
    uint8_t* p
    Py_ssize_t h
    Py_ssize_t w

# A CSR op table: op i covers entries start[i]..start[i+1]. ext holds each
# op's inclusive (y_lo, y_hi, x_lo, x_hi), y_lo > y_hi when empty. sy/sx/uv
# and fstart/fac are NULL where a table has no such column.
ctypedef struct Ops:
    const int32_t* dy
    const int32_t* dx
    const int32_t* sy
    const int32_t* sx
    const int32_t* uv
    const int64_t* start
    const int32_t* ext
    const int64_t* fstart
    const float* fac

# One tile's colour source: a texture crop at (oy, ox), or a flat colour.
ctypedef struct Src:
    const uint8_t* tex
    Py_ssize_t stride
    Py_ssize_t ox
    Py_ssize_t oy
    const uint8_t* flat

ctypedef struct Stamps:
    const int32_t* dy
    const int32_t* dx
    const uint16_t* tgt
    const uint16_t* alpha
    const int64_t* start
    const int32_t* ext

ctypedef struct Units:
    const int32_t* index
    const int32_t* rows
    const uint64_t* addrs
    const int32_t* dims
    const uint8_t* farm_mask
    const uint8_t* farm_rgb

# Unit-pack row kinds (native_composite.UnitPack).
DEF ROW_MARK = 0
DEF ROW_CONFORM = 1
DEF ROW_SPRITE = 2

# Stepped op-table layout (native_composite.stepped_tables).
DEF OP_DIAMOND = 0
DEF OP_SEAM_UL = 1
DEF OP_SEAM_UR = 2
DEF OP_SEAM_APEX = 3
DEF OP_EDGE0 = 4
DEF N_FIXED = 8
DEF PER_RISE = 9


cdef inline int _clip_state(const int32_t* e, Py_ssize_t by, Py_ssize_t bx, const Canvas* c) noexcept nogil:
    """0 wholly outside (or empty), 1 wholly inside, 2 straddling an edge:
    render._clipped_paint's extent pre-check."""
    if e[0] > e[1]:
        return 0
    if by + e[1] < 0 or by + e[0] >= c.h or bx + e[3] < 0 or bx + e[2] >= c.w:
        return 0
    if by + e[0] >= 0 and by + e[1] < c.h and bx + e[2] >= 0 and bx + e[3] < c.w:
        return 1
    return 2


cdef void _op_scatter(const Canvas* c, const Ops* t, Py_ssize_t op, Py_ssize_t by, Py_ssize_t bx,
                      const Src* s, const uint8_t* lut) noexcept nogil:
    cdef int st = _clip_state(t.ext + 4 * op, by, bx, c)
    cdef Py_ssize_t k, py, px
    cdef uint8_t* d
    cdef const uint8_t* q
    if st == 0:
        return
    for k in range(t.start[op], t.start[op + 1]):
        py = by + t.dy[k]
        px = bx + t.dx[k]
        if st == 2 and (py < 0 or py >= c.h or px < 0 or px >= c.w):
            continue
        d = c.p + (py * c.w + px) * 3
        if s.tex != NULL:
            q = s.tex + (s.oy + t.sy[k]) * s.stride + (s.ox + t.sx[k]) * 3
        else:
            q = s.flat
        if lut != NULL:
            d[0] = lut[q[0]]
            d[1] = lut[q[1]]
            d[2] = lut[q[2]]
        else:
            d[0] = q[0]
            d[1] = q[1]
            d[2] = q[2]


cdef void _op_darken(const Canvas* c, const Ops* t, Py_ssize_t op, Py_ssize_t by, Py_ssize_t bx) noexcept nogil:
    cdef int st = _clip_state(t.ext + 4 * op, by, bx, c)
    cdef Py_ssize_t k, py, px, s0 = t.start[op], f0 = t.fstart[op]
    cdef uint8_t* d
    cdef float f
    if st == 0:
        return
    for k in range(s0, t.start[op + 1]):
        py = by + t.dy[k]
        px = bx + t.dx[k]
        if st == 2 and (py < 0 or py >= c.h or px < 0 or px >= c.w):
            continue
        d = c.p + (py * c.w + px) * 3
        f = t.fac[f0 + k - s0]
        d[0] = <uint8_t>(<float>d[0] * f)
        d[1] = <uint8_t>(<float>d[1] * f)
        d[2] = <uint8_t>(<float>d[2] * f)


cdef void _op_solid(const Canvas* c, const Ops* t, Py_ssize_t op, Py_ssize_t by, Py_ssize_t bx,
                    uint8_t r, uint8_t g, uint8_t b) noexcept nogil:
    cdef int st = _clip_state(t.ext + 4 * op, by, bx, c)
    cdef Py_ssize_t k, py, px
    cdef uint8_t* d
    if st == 0:
        return
    for k in range(t.start[op], t.start[op + 1]):
        py = by + t.dy[k]
        px = bx + t.dx[k]
        if st == 2 and (py < 0 or py >= c.h or px < 0 or px >= c.w):
            continue
        d = c.p + (py * c.w + px) * 3
        d[0] = r
        d[1] = g
        d[2] = b


cdef void _op_shade(const Canvas* c, const Ops* t, Py_ssize_t op, Py_ssize_t by, Py_ssize_t bx,
                    const Src* s, const float* shade) noexcept nogil:
    """render._render_tile_sloped's paint: float32 multiply, clip, truncate."""
    cdef int st = _clip_state(t.ext + 4 * op, by, bx, c)
    cdef Py_ssize_t k, py, px, ch
    cdef uint8_t* d
    cdef const uint8_t* q
    cdef float sv, v
    if st == 0:
        return
    for k in range(t.start[op], t.start[op + 1]):
        py = by + t.dy[k]
        px = bx + t.dx[k]
        if st == 2 and (py < 0 or py >= c.h or px < 0 or px >= c.w):
            continue
        d = c.p + (py * c.w + px) * 3
        if s.tex != NULL:
            q = s.tex + (s.oy + t.sy[k]) * s.stride + (s.ox + t.sx[k]) * 3
        else:
            q = s.flat
        sv = shade[t.uv[k]]
        for ch in range(3):
            v = <float>q[ch] * sv
            if v < 0:
                v = 0
            elif v > 255:
                v = 255
            d[ch] = <uint8_t>v


cdef void _stamp(const Canvas* c, const Stamps* g, Py_ssize_t sid, Py_ssize_t by, Py_ssize_t bx) noexcept nogil:
    """render._clipped_lerp of one baked grid stamp."""
    cdef int st = _clip_state(g.ext + 4 * sid, by, bx, c)
    cdef Py_ssize_t k, py, px, ch
    cdef uint8_t* d
    cdef unsigned int a
    if st == 0:
        return
    for k in range(g.start[sid], g.start[sid + 1]):
        py = by + g.dy[k]
        px = bx + g.dx[k]
        if st == 2 and (py < 0 or py >= c.h or px < 0 or px >= c.w):
            continue
        d = c.p + (py * c.w + px) * 3
        a = g.alpha[k]
        for ch in range(3):
            d[ch] = <uint8_t>((g.tgt[3 * k + ch] * a + d[ch] * (255 - a)) // 255)


cdef void _blit(const Canvas* c, const uint8_t* rgba, Py_ssize_t h, Py_ssize_t w,
                Py_ssize_t base_y, Py_ssize_t base_x) noexcept nogil:
    """render._clipped_paint_rgba on a contiguous (h, w, 4) block."""
    cdef Py_ssize_t sy0, sx0, sy1, sx1, y, x, ch
    cdef const uint8_t* q
    cdef uint8_t* d
    cdef unsigned int a
    sy0 = -base_y if base_y < 0 else 0
    sx0 = -base_x if base_x < 0 else 0
    sy1 = h if h < c.h - base_y else c.h - base_y
    sx1 = w if w < c.w - base_x else c.w - base_x
    if sy0 >= sy1 or sx0 >= sx1:
        return
    for y in range(sy0, sy1):
        q = rgba + (y * w + sx0) * 4
        d = c.p + ((base_y + y) * c.w + base_x + sx0) * 3
        for x in range(sx0, sx1):
            a = q[3]
            # The blend's exact values at both ends: d at 0, the source at 255.
            if a == 255:
                d[0] = q[0]
                d[1] = q[1]
                d[2] = q[2]
            elif a != 0:
                for ch in range(3):
                    d[ch] = <uint8_t>((q[ch] * a + d[ch] * (255 - a)) // 255)
            q += 4
            d += 3


cdef void _farm_and_units(const Canvas* c, const Units* u, Py_ssize_t t, const Ops* edges, Py_ssize_t edge0,
                          const Stamps* g, Py_ssize_t sid, const Ops* marks, Py_ssize_t mark_op,
                          const Ops* conform, Py_ssize_t conform_op,
                          Py_ssize_t by, Py_ssize_t bx, Py_ssize_t off_x, Py_ssize_t off_y) noexcept nogil:
    """Everything after a tile's terrain, in _paint_tile_and_units_*'s order:
    farm outline, grid, marks, then sprites."""
    cdef Py_ssize_t j, i, r0, slot
    cdef uint8_t mask
    cdef const uint8_t* rgb
    cdef const int32_t* row
    cdef int32_t packed
    if u != NULL:
        mask = u.farm_mask[t]
        if mask:
            rgb = u.farm_rgb + 3 * t
            for j in range(4):
                if mask & (1 << j):
                    _op_solid(c, edges, edge0 + j, by, bx, rgb[0], rgb[1], rgb[2])
    if sid >= 0:
        _stamp(c, g, sid, by, bx)
    if u == NULL:
        return
    r0 = u.index[2 * t]
    for i in range(r0, r0 + u.index[2 * t + 1]):
        row = u.rows + 4 * i
        if row[0] == ROW_SPRITE:
            slot = row[3]
            _blit(c, <const uint8_t*><uintptr_t>u.addrs[slot], u.dims[2 * slot], u.dims[2 * slot + 1],
                  row[1] - off_y, row[2] - off_x)
            continue
        packed = row[3]
        if row[0] == ROW_CONFORM:
            _op_solid(c, conform, conform_op, by, bx,
                      (packed >> 16) & 255, (packed >> 8) & 255, packed & 255)
        else:
            _op_solid(c, marks, mark_op, row[1] - off_y, row[2] - off_x,
                      (packed >> 16) & 255, (packed >> 8) & 255, packed & 255)


cdef inline const int32_t* _p32(const int32_t[::1] a):
    return &a[0] if a.shape[0] else NULL


cdef inline const int64_t* _p64(const int64_t[::1] a):
    return &a[0] if a.shape[0] else NULL


cdef inline const float* _pf(const float[::1] a):
    return &a[0] if a.shape[0] else NULL


cdef Ops _ops(tuple cols):
    """(dy, dx, sy, sx, uv, start, ext, fstart, fac), any of the optional
    columns None."""
    cdef Ops t
    cdef const int32_t[::1] v32
    cdef const int64_t[::1] v64
    cdef const float[::1] vf
    t.dy = _p32(cols[0])
    t.dx = _p32(cols[1])
    t.sy = NULL if cols[2] is None else _p32(cols[2])
    t.sx = NULL if cols[3] is None else _p32(cols[3])
    t.uv = NULL if cols[4] is None else _p32(cols[4])
    t.start = _p64(cols[5])
    v32 = cols[6]
    t.ext = &v32[0]
    t.fstart = NULL if cols[7] is None else _p64(cols[7])
    t.fac = NULL if cols[8] is None else _pf(cols[8])
    return t


cdef Stamps _stamps(tuple cols):
    cdef Stamps g
    cdef const uint16_t[::1] tgt
    cdef const uint16_t[::1] alpha
    cdef const int32_t[::1] ext
    g.dy = _p32(cols[0])
    g.dx = _p32(cols[1])
    tgt = cols[2]
    alpha = cols[3]
    g.tgt = &tgt[0] if tgt.shape[0] else NULL
    g.alpha = &alpha[0] if alpha.shape[0] else NULL
    g.start = _p64(cols[4])
    ext = cols[5]
    g.ext = &ext[0] if ext.shape[0] else NULL
    return g


cdef Units _units(tuple cols):
    cdef Units u
    cdef const uint64_t[::1] addrs
    cdef const uint8_t[::1] mask
    cdef const uint8_t[::1] rgb
    u.index = _p32(cols[0])
    u.rows = _p32(cols[1])
    addrs = cols[2]
    u.addrs = &addrs[0] if addrs.shape[0] else NULL
    u.dims = _p32(cols[3])
    mask = cols[4]
    rgb = cols[5]
    u.farm_mask = &mask[0]
    u.farm_rgb = &rgb[0]
    return u


cdef Src* _sources(list textures, const uint8_t[:, ::1] flat) except NULL:
    """One Src per texture slot: a contiguous (T, T, 3) texture, or None for
    that slot's flat colour. The caller's list keeps every array alive."""
    cdef Py_ssize_t i, n = len(textures)
    cdef const uint8_t[:, :, ::1] tv
    cdef Src* out = <Src*>malloc(max(n, 1) * sizeof(Src))
    if out == NULL:
        raise MemoryError()
    for i in range(n):
        out[i].flat = &flat[i, 0]
        out[i].ox = 0
        out[i].oy = 0
        if textures[i] is None:
            out[i].tex = NULL
            out[i].stride = 0
        else:
            tv = textures[i]
            out[i].tex = &tv[0, 0, 0]
            out[i].stride = tv.shape[1] * 3
    return out


def composite_iso(uint8_t[:, :, ::1] img, Py_ssize_t off_x, Py_ssize_t off_y,
                  const int64_t[::1] cand_x, const int64_t[::1] cand_y,
                  const int32_t[::1] cand_tex, const int32_t[::1] cand_grid,
                  const int64_t[:, :] elev, tuple geom, tuple tables, tuple textures,
                  tuple stamps, tuple units):
    """composite_rect_iso's per-tile loop over the ordered candidates.
    geom is (origin_x, origin_y, half_w, half_h, elev_step, tile_px,
    max_rise). Returns False, having painted nothing, when a neighbour rise
    is past the tables' max_rise; the caller then takes the per-tile path."""
    cdef Py_ssize_t origin_x = geom[0], origin_y = geom[1], half_w = geom[2], half_h = geom[3]
    cdef Py_ssize_t step = geom[4], tile_px = geom[5], max_rise = geom[6]
    cdef Py_ssize_t n = cand_x.shape[0], mh = elev.shape[0], mw = elev.shape[1]
    cdef Py_ssize_t k, tx, ty, own, dl, dr, rul, rur, rdg, bx, by, base, slot, tsize
    cdef bint seam, bad = False
    cdef Canvas c
    cdef Ops t = _ops(tables[:9])
    cdef const uint8_t[:, ::1] luts = tables[9]
    cdef const int64_t[::1] tex_size = textures[2]
    cdef Stamps g
    cdef Units u
    cdef const Units* up = NULL
    cdef Src* srcs
    cdef Src s
    if img.shape[2] != 3 or luts.shape[0] != 2 or luts.shape[1] != 256:
        raise ValueError("img must have 3 channels and luts shape (2, 256)")
    if not (cand_y.shape[0] == n and cand_tex.shape[0] == n and cand_grid.shape[0] == n):
        raise ValueError("candidate arrays differ in length")
    c.p = &img[0, 0, 0]
    c.h = img.shape[0]
    c.w = img.shape[1]
    if stamps is not None:
        g = _stamps(stamps)
    if units is not None:
        u = _units(units)
        up = &u
    srcs = _sources(textures[0], textures[1])
    try:
        with nogil:
            for k in range(n):
                tx = cand_x[k]
                ty = cand_y[k]
                own = elev[ty, tx]
                if ((tx > 0 and own - elev[ty, tx - 1] > max_rise) or (ty + 1 < mh and own - elev[ty + 1, tx] > max_rise)
                        or (ty > 0 and own - elev[ty - 1, tx] > max_rise) or (tx + 1 < mw and own - elev[ty, tx + 1] > max_rise)
                        or (tx + 1 < mw and ty > 0 and own - elev[ty - 1, tx + 1] > max_rise)):
                    bad = True
                    break
            if not bad:
                for k in range(n):
                    tx = cand_x[k]
                    ty = cand_y[k]
                    own = elev[ty, tx]
                    bx = origin_x + (tx + ty) * half_w - off_x
                    by = origin_y + (ty - tx) * half_h - own * step - off_y
                    dl = max(own - elev[ty, tx - 1], 0) if tx > 0 else 0
                    dr = max(own - elev[ty + 1, tx], 0) if ty + 1 < mh else 0
                    rul = max(own - elev[ty - 1, tx], 0) if ty > 0 else 0
                    rur = max(own - elev[ty, tx + 1], 0) if tx + 1 < mw else 0
                    rdg = max(own - elev[ty - 1, tx + 1], 0) if tx + 1 < mw and ty > 0 else 0

                    slot = cand_tex[k]
                    s = srcs[slot]
                    if s.tex != NULL:
                        tsize = tex_size[slot]
                        s.ox = (tx * tile_px) % tsize
                        s.oy = (ty * tile_px) % tsize
                    if dl > 0:
                        _op_scatter(&c, &t, N_FIXED + (dl - 1) * PER_RISE + 0, by, bx, &s, &luts[0, 0])
                    if dr > 0:
                        _op_scatter(&c, &t, N_FIXED + (dr - 1) * PER_RISE + 1, by, bx, &s, &luts[1, 0])
                    _op_scatter(&c, &t, OP_DIAMOND, by, bx, &s, NULL)

                    seam = rul > 0 or rur > 0
                    if rul > 0:
                        _op_darken(&c, &t, OP_SEAM_UL, by, bx)
                    if rur > 0:
                        _op_darken(&c, &t, OP_SEAM_UR, by, bx)
                    if seam:
                        _op_darken(&c, &t, OP_SEAM_APEX, by, bx)
                    if rul > 0:
                        _op_darken(&c, &t, N_FIXED + (rul - 1) * PER_RISE + 2, by, bx)
                    if rur > 0:
                        _op_darken(&c, &t, N_FIXED + (rur - 1) * PER_RISE + 3, by, bx)
                    if seam and rdg > 0:
                        base = N_FIXED + (rdg - 1) * PER_RISE
                        if rul > 0 and rur > 0:
                            _op_darken(&c, &t, base + 4, by, bx)
                        elif rul > 0:
                            _op_darken(&c, &t, base + 5, by, bx)
                        else:
                            _op_darken(&c, &t, base + 6, by, bx)
                    if tx > 0 and rul > 0 and elev[ty - 1, tx - 1] > elev[ty - 1, tx]:
                        _op_darken(&c, &t, N_FIXED + (rul - 1) * PER_RISE + 7, by, bx)
                    if ty + 1 < mh and rur > 0 and elev[ty + 1, tx + 1] > elev[ty, tx + 1]:
                        _op_darken(&c, &t, N_FIXED + (rur - 1) * PER_RISE + 8, by, bx)

                    _farm_and_units(&c, up, ty * mw + tx, &t, OP_EDGE0, &g, cand_grid[k],
                                    &t, OP_DIAMOND, NULL, 0, by, bx, off_x, off_y)
    finally:
        free(srcs)
    return not bad


def composite_sloped(uint8_t[:, :, ::1] img, Py_ssize_t off_x, Py_ssize_t off_y,
                     const int64_t[::1] cand_x, const int64_t[::1] cand_y,
                     const int32_t[::1] cand_tex, const int32_t[::1] cand_grid,
                     const int32_t[::1] cand_shape, const int64_t[::1] cand_shade,
                     const int64_t[:, :] corner_rise, tuple geom, tuple quads, tuple edges,
                     tuple diamond, const float[::1] shade, tuple textures, tuple stamps, tuple units):
    """composite_rect_sloped's per-tile loop. geom is (origin_x, origin_y,
    half_w, half_h, tile_px); cand_shape indexes quads (one op per shape) and
    edges (four per shape), cand_shade is each candidate's offset into
    shade."""
    cdef Py_ssize_t origin_x = geom[0], origin_y = geom[1], half_w = geom[2], half_h = geom[3]
    cdef Py_ssize_t tile_px = geom[4]
    cdef Py_ssize_t n = cand_x.shape[0], mw = corner_rise.shape[1] - 1
    cdef Py_ssize_t k, tx, ty, d_nw, d_ne, d_sw, d_se, dmin, bx, by, slot, tsize, shp
    cdef Canvas c
    cdef Ops q = _ops(quads)
    cdef Ops e = _ops(edges)
    cdef Ops dm = _ops(diamond)
    cdef const int64_t[::1] tex_size = textures[2]
    cdef Stamps g
    cdef Units u
    cdef const Units* up = NULL
    cdef Src* srcs
    cdef Src s
    if img.shape[2] != 3:
        raise ValueError("img must have 3 channels")
    if not (cand_y.shape[0] == n and cand_tex.shape[0] == n and cand_grid.shape[0] == n
            and cand_shape.shape[0] == n and cand_shade.shape[0] == n):
        raise ValueError("candidate arrays differ in length")
    c.p = &img[0, 0, 0]
    c.h = img.shape[0]
    c.w = img.shape[1]
    if stamps is not None:
        g = _stamps(stamps)
    if units is not None:
        u = _units(units)
        up = &u
    srcs = _sources(textures[0], textures[1])
    try:
        with nogil:
            for k in range(n):
                tx = cand_x[k]
                ty = cand_y[k]
                d_nw = corner_rise[ty, tx]
                d_ne = corner_rise[ty, tx + 1]
                d_sw = corner_rise[ty + 1, tx]
                d_se = corner_rise[ty + 1, tx + 1]
                dmin = min(min(d_nw, d_ne), min(d_sw, d_se))
                bx = origin_x + (tx + ty) * half_w - off_x
                by = origin_y + (ty - tx) * half_h - off_y - dmin
                slot = cand_tex[k]
                s = srcs[slot]
                if s.tex != NULL:
                    tsize = tex_size[slot]
                    s.ox = (tx * tile_px) % tsize
                    s.oy = (ty * tile_px) % tsize
                shp = cand_shape[k]
                _op_shade(&c, &q, shp, by, bx, &s, &shade[cand_shade[k]])
                _farm_and_units(&c, up, ty * mw + tx, &e, 4 * shp, &g, cand_grid[k],
                                &dm, 0, &q, shp, by, bx, off_x, off_y)
    finally:
        free(srcs)


# ---------------------------------------------------------------------------
# SLD: descape/sld_decoder.py's per-file walk and per-layer delta-chain decode.
# Both return a failure sentinel wherever the Python code would raise; the
# caller then re-runs the Python code to raise the exact SLDError.

# SLDFile's flat index record widths (sld_decoder._LAYER_FIELDS/_FRAME_FIELDS).
DEF SLD_LAYER_FIELDS = 11
DEF SLD_FRAME_FIELDS = 8
DEF SLD_KINDS = 5
DEF SLD_DELTA_FLAG = 0x80
DEF SLD_BLOCK_BYTES = 8


cdef inline uint32_t _u16(const uint8_t* p) noexcept nogil:
    return p[0] | (<uint32_t>p[1] << 8)


cdef inline uint32_t _u32(const uint8_t* p) noexcept nogil:
    return p[0] | (<uint32_t>p[1] << 8) | (<uint32_t>p[2] << 16) | (<uint32_t>p[3] << 24)


def sld_walk(const uint8_t[::1] data, Py_ssize_t layout_tag, Py_ssize_t frame_count,
             uint32_t[::1] frames, uint32_t[::1] layers, uint32_t[:, ::1] chains, int64_t[::1] chain_len):
    """SLDFile._walk's frame loop, after its header checks. frames holds
    frame_count records, layers 4 per frame (OUTLINE is never recorded),
    chains (5, frame_count). Returns the layer count, or -1 where the Python
    walk would raise."""
    cdef int64_t size = data.shape[0], result
    if layout_tag < 0 or frame_count < 0:
        raise ValueError("layout_tag and frame_count must be non-negative")
    if frames.shape[0] < frame_count * SLD_FRAME_FIELDS or layers.shape[0] < frame_count * 4 * SLD_LAYER_FIELDS:
        raise ValueError("output buffers too small for frame_count")
    if chains.shape[0] != SLD_KINDS or chains.shape[1] < frame_count or chain_len.shape[0] != SLD_KINDS:
        raise ValueError("chains must be (5, >= frame_count) and chain_len 5 long")
    if frame_count == 0:
        chain_len[:] = 0
        return 0
    with nogil:
        result = _sld_walk(&data[0] if size else NULL, size, layout_tag, frame_count,
                           &frames[0], &layers[0], &chains[0, 0], chains.shape[1], &chain_len[0])
    return result


cdef int64_t _sld_walk(const uint8_t* d, int64_t size, int64_t layout_tag, int64_t frame_count,
                       uint32_t* frames, uint32_t* layers, uint32_t* chains, int64_t chain_stride,
                       int64_t* chain_len) noexcept nogil:
    cdef int64_t offset = layout_tag, start, length, command_offset
    cdef int64_t layer_id = 0, first_layer, position
    cdef uint32_t width, height, hotspot_x, hotspot_y, frame_type, frame_index
    cdef uint32_t x1, y1, x2, y2, flag0, flag1, command_count
    cdef uint32_t main_x1, main_y1, main_w, main_h, bx1, by1, bw, bh
    cdef int kind
    cdef uint32_t* rec
    for kind in range(SLD_KINDS):
        chain_len[kind] = 0
    for position in range(frame_count):
        if offset + 12 > size:
            return -1
        width = _u16(d + offset)
        height = _u16(d + offset + 2)
        hotspot_x = _u16(d + offset + 4)
        hotspot_y = _u16(d + offset + 6)
        frame_type = d[offset + 8]
        frame_index = _u16(d + offset + 10)
        offset += 12
        first_layer = layer_id
        main_x1 = main_y1 = main_w = main_h = 0
        for kind in range(SLD_KINDS):
            if not (frame_type & (1 << kind)):
                continue
            start = offset
            if kind == 0 or kind == 1:
                if start + 16 > size:
                    return -1
                length = _u32(d + start)
                if length < 4 or start + length > size:
                    return -1
                x1 = _u16(d + start + 4)
                y1 = _u16(d + start + 6)
                x2 = _u16(d + start + 8)
                y2 = _u16(d + start + 10)
                flag0 = d[start + 12]
                flag1 = d[start + 13]
                command_count = _u16(d + start + 14)
                if x2 < x1 or y2 < y1:
                    return -1
                bx1, by1, bw, bh = x1, y1, x2 - x1, y2 - y1
                if kind == 0:
                    main_x1, main_y1, main_w, main_h = bx1, by1, bw, bh
                command_offset = start + 16
            elif kind == 2:
                if start + 4 > size:
                    return -1
                length = _u32(d + start)
                if length < 4 or start + length > size:
                    return -1
                offset = start + length
                offset += (layout_tag - offset) & 3
                continue
            else:
                if start + 8 > size:
                    return -1
                length = _u32(d + start)
                if length < 4 or start + length > size:
                    return -1
                flag0 = d[start + 4]
                flag1 = d[start + 5]
                command_count = _u16(d + start + 6)
                bx1, by1, bw, bh = main_x1, main_y1, main_w, main_h
                command_offset = start + 8
            if command_offset + 2 * <int64_t>command_count > size:
                return -1
            rec = &layers[layer_id * SLD_LAYER_FIELDS]
            rec[0] = kind
            rec[1] = bx1
            rec[2] = by1
            rec[3] = bw
            rec[4] = bh
            rec[5] = flag0
            rec[6] = flag1
            rec[7] = frame_index
            rec[8] = command_count
            rec[9] = <uint32_t>command_offset
            rec[10] = <uint32_t>chain_len[kind]
            chains[kind * chain_stride + chain_len[kind]] = <uint32_t>layer_id
            chain_len[kind] += 1
            layer_id += 1
            offset = start + length
            offset += (layout_tag - offset) & 3
        rec = &frames[position * SLD_FRAME_FIELDS]
        rec[0] = width
        rec[1] = height
        rec[2] = hotspot_x
        rec[3] = hotspot_y
        rec[4] = frame_type
        rec[5] = frame_index
        rec[6] = <uint32_t>first_layer
        rec[7] = <uint32_t>(layer_id - first_layer)
    return layer_id


cdef inline int64_t _floor_div4(int64_t a) noexcept nogil:
    """Python's a // 4: cdivision truncates toward zero instead."""
    cdef int64_t q = a / 4
    if a % 4 != 0 and a < 0:
        q -= 1
    return q


cdef inline void _bc1_block(const uint8_t* raw, uint8_t* out) noexcept nogil:
    """sld_decoder.decode_bc1_blocks for one block: 16 RGBA pixels, row-major."""
    cdef int c0[3]
    cdef int c1[3]
    cdef uint8_t pal[16]
    cdef bint opaque = _u16(raw) > _u16(raw + 2)
    cdef int ch, p, idx
    c0[0] = ((raw[1] & 0xF8) >> 3) * 8
    c0[1] = (((raw[1] & 0x07) << 3) + ((raw[0] & 0xE0) >> 5)) * 4
    c0[2] = (raw[0] & 0x1F) * 8
    c1[0] = ((raw[3] & 0xF8) >> 3) * 8
    c1[1] = (((raw[3] & 0x07) << 3) + ((raw[2] & 0xE0) >> 5)) * 4
    c1[2] = (raw[2] & 0x1F) * 8
    for ch in range(3):
        pal[ch] = <uint8_t>c0[ch]
        pal[4 + ch] = <uint8_t>c1[ch]
        if opaque:
            pal[8 + ch] = <uint8_t>((2 * c0[ch] + c1[ch] + 1) / 3)
            pal[12 + ch] = <uint8_t>((c0[ch] + 2 * c1[ch] + 1) / 3)
        else:
            pal[8 + ch] = <uint8_t>((c0[ch] + c1[ch]) / 2)
            pal[12 + ch] = 0
    pal[3] = 255
    pal[7] = 255
    pal[11] = 255
    pal[15] = 255 if opaque else 0
    for p in range(16):
        idx = (raw[4 + (p >> 2)] >> (2 * (p & 3))) & 3
        memcpy(out + 4 * p, pal + 4 * idx, 4)


cdef inline void _bc4_block(const uint8_t* raw, uint8_t* out) noexcept nogil:
    """sld_decoder.decode_bc4_blocks for one block: the value in R, alpha in A."""
    cdef int a0 = raw[0], a1 = raw[1], step, p, idx
    cdef bint wide = a0 > a1
    cdef uint8_t values[8]
    cdef uint8_t alpha[8]
    cdef uint32_t group
    values[0] = <uint8_t>a0
    values[1] = <uint8_t>a1
    for step in range(1, 6):
        if wide:
            values[step + 1] = <uint8_t>(((7 - step) * a0 + step * a1) / 7)
        elif step < 5:
            values[step + 1] = <uint8_t>(((5 - step) * a0 + step * a1) / 5)
        else:
            values[step + 1] = 0
    values[7] = <uint8_t>((a0 + 6 * a1) / 7) if wide else 255
    for p in range(8):
        alpha[p] = 255
    if not wide:
        alpha[6] = 0
    for p in range(16):
        if p < 8:
            group = raw[2] | (<uint32_t>raw[3] << 8) | (<uint32_t>raw[4] << 16)
            idx = (group >> (3 * p)) & 7
        else:
            group = raw[5] | (<uint32_t>raw[6] << 8) | (<uint32_t>raw[7] << 16)
            idx = (group >> (3 * (p - 8))) & 7
        out[4 * p] = values[idx]
        out[4 * p + 1] = 0
        out[4 * p + 2] = 0
        out[4 * p + 3] = alpha[idx]


cdef bint _decode_link(const uint8_t* d, int64_t size, const uint32_t* rec, bint bc1,
                       uint8_t* out, const uint8_t* prev, const uint32_t* prev_rec) noexcept nogil:
    """SLDFile._decode_blocks for one link into out, (block_count, 16, 4).
    prev is the previous link's blocks, or NULL for a chain's first link.
    False where _decode_blocks would raise."""
    cdef int64_t wb = rec[3] / 4, hb = rec[4] / 4, block_count = wb * hb
    cdef int64_t command_count = rec[8], command_offset = rec[9]
    cdef int64_t block_offset = command_offset + 2 * command_count
    cdef int64_t filled = 0, drawn = 0, position = 0, k, dest, n, x, y
    cdef int64_t pwb = 0, phb = 0, dx = 0, dy = 0
    cdef bint delta = prev != NULL and (rec[5] & SLD_DELTA_FLAG) != 0 and rec[7] > 0
    cdef uint8_t skip, draw
    for k in range(command_count):
        filled += d[command_offset + 2 * k] + d[command_offset + 2 * k + 1]
        drawn += d[command_offset + 2 * k + 1]
    if filled > block_count:
        return False
    if drawn and block_offset + SLD_BLOCK_BYTES * drawn > size:
        return False
    memset(out, 0, block_count * 64)
    if delta:
        pwb = prev_rec[3] / 4
        phb = prev_rec[4] / 4
        dx = _floor_div4(<int64_t>rec[1] - <int64_t>prev_rec[1])
        dy = _floor_div4(<int64_t>rec[2] - <int64_t>prev_rec[2])
        if wb == 0 or pwb == 0 or phb == 0:
            delta = False
    drawn = 0
    for k in range(command_count):
        skip = d[command_offset + 2 * k]
        draw = d[command_offset + 2 * k + 1]
        if delta:
            for dest in range(position, position + skip):
                x = dest % wb + dx
                y = dest / wb + dy
                if 0 <= x < pwb and 0 <= y < phb:
                    memcpy(out + 64 * dest, prev + 64 * (x + y * pwb), 64)
        position += skip
        for n in range(draw):
            if bc1:
                _bc1_block(d + block_offset + SLD_BLOCK_BYTES * drawn, out + 64 * position)
            else:
                _bc4_block(d + block_offset + SLD_BLOCK_BYTES * drawn, out + 64 * position)
            drawn += 1
            position += 1
    return True


def sld_decode_layer(const uint8_t[::1] data, const uint32_t[::1] layers, const uint32_t[::1] chain,
                     Py_ssize_t layer_id, bint bc1, uint8_t[:, :, ::1] canvas):
    """SLDFile._decode_layer_blocks, _blocks_to_image and _paste_on_canvas for
    one layer: resolves its delta chain back to the nearest non-delta link,
    decodes forward and pastes the result, clipped, onto the zeroed (h, w, 4)
    canvas. False, with canvas unspecified, where the Python would raise."""
    cdef int64_t size = data.shape[0], n_layers = layers.shape[0] // SLD_LAYER_FIELDS
    cdef const uint8_t* d = &data[0] if size else NULL
    cdef const uint32_t* rec
    cdef const uint32_t* prev_rec = NULL
    cdef int64_t position, root, p, wb, hb, most = 0, iy, ix, copy_h, copy_w, x0, y0, by, bx
    cdef uint8_t* cur
    cdef uint8_t* prev
    cdef uint8_t* tmp
    cdef bint ok = True
    if not 0 <= layer_id < n_layers or canvas.shape[2] != 4:
        raise ValueError("layer_id outside layers, or canvas not (h, w, 4)")
    rec = &layers[layer_id * SLD_LAYER_FIELDS]
    position = rec[10]
    if position >= chain.shape[0] or chain[position] != layer_id:
        raise ValueError("chain does not hold layer_id at its chain position")
    root = position
    while root > 0:
        rec = &layers[chain[root] * SLD_LAYER_FIELDS]
        if not ((rec[5] & SLD_DELTA_FLAG) and rec[7] > 0):
            break
        root -= 1
    for p in range(root, position + 1):
        rec = &layers[chain[p] * SLD_LAYER_FIELDS]
        if (rec[3] / 4) * (rec[4] / 4) > most:
            most = (rec[3] / 4) * (rec[4] / 4)
    cur = <uint8_t*>malloc(max(most, 1) * 64)
    prev = <uint8_t*>malloc(max(most, 1) * 64)
    if cur == NULL or prev == NULL:
        free(cur)
        free(prev)
        raise MemoryError()
    try:
        with nogil:
            for p in range(root, position + 1):
                rec = &layers[chain[p] * SLD_LAYER_FIELDS]
                if not _decode_link(d, size, rec, bc1, cur, prev if p > root else NULL, prev_rec):
                    ok = False
                    break
                prev_rec = rec
                tmp = prev
                prev = cur
                cur = tmp
            if ok:
                # prev now holds the last link's blocks; rec is that layer.
                wb = rec[3] / 4
                hb = rec[4] / 4
                x0 = rec[1]
                y0 = rec[2]
                copy_h = min(hb * 4, canvas.shape[0] - y0)
                copy_w = min(wb * 4, canvas.shape[1] - x0)
                for iy in range(copy_h):
                    by = iy >> 2
                    for ix in range(copy_w):
                        bx = ix >> 2
                        memcpy(&canvas[y0 + iy, x0 + ix, 0], prev + 64 * (by * wb + bx) + 4 * (4 * (iy & 3) + (ix & 3)), 4)
    finally:
        free(cur)
        free(prev)
    return ok
