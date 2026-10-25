# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Native composite kernel. Built by tools/build_native.py; selected by
descape/composite_backend.py, which falls back to numpy when this is missing
or its KERNEL_ABI doesn't match. Every function must stay byte-identical to
the numpy code it replaces (tests/test_native_composite.py,
tests/test_terrain_blend_render.py for blended top blocks,
tests/test_sld_native.py for the SLD walk and decode, and
tests/test_unit_sprites.py for the sprite tint and box shrink at the end).

Arithmetic mirrors numpy's dtypes exactly: opaque scatters are uint8, darken
and slope shade are float32 multiplies truncated to uint8 (never a double
temporary), and the lerp and rgba blit are integer (t*a + d*(255-a)) // 255.
No index producer repeats a destination within one call, so writing in place
reads each pixel once, the same as numpy's fancy-index read-modify-write."""

from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, uintptr_t
from libc.stdlib cimport calloc, free, malloc
from libc.string cimport memcpy, memset

# Bump on any signature or semantics change, together with
# composite_backend.EXPECTED_KERNEL_ABI, so a stale build is never used.
KERNEL_ABI = 10


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
    list order (seams, seam apex, shadow bands, apex wedge). The caller
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
# render._paint_candidates_iso/_sloped exactly: one interleaved loop with
# sprites off, terrain then objects (two_pass) with them on.

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

# Terrain blending (TASK-143): per candidate a spec id or -1, each spec a run
# of (kind, texture slot, mask id) ops in terrain_blend._compose's order.
# block is the per-call scratch a blended top block is composed into, at the
# n_samp (samp_y, samp_x) texels only (TASK-031.66), zeroed once per call.
# Invariant: the samples are the union of every Src-reading op's (sy, sx),
# so a new op that reads a Src must join SteppedTables/SlopedShapes' union.
ctypedef struct Blend:
    const int32_t* cand
    const int64_t* start
    const int32_t* ops
    const uint8_t** masks
    Py_ssize_t* mask_stride
    uint8_t* block
    const int32_t* samp_y
    const int32_t* samp_x
    Py_ssize_t n_samp

# Blend op kinds (native_composite.BLEND_LAYER/BLEND_NEIGHBOUR).
DEF BLEND_LAYER = 0
DEF BLEND_NEIGHBOUR = 1

# Unit-pack rows (native_composite.UnitPack): ROW_W int32s, (kind, y, x,
# slot or rgb, base rise).
DEF ROW_MARK = 0
DEF ROW_CONFORM = 1
DEF ROW_SPRITE = 2
DEF ROW_W = 5

# render.INK_NONE: "no ink" in a two-pass ink plane, above every rise.
DEF INK_NONE = 0x7FFFFFFF

# Stepped op-table layout (native_composite.stepped_tables).
DEF OP_DIAMOND = 0
DEF OP_SEAM_UL = 1
DEF OP_SEAM_UR = 2
DEF OP_SEAM_APEX = 3
DEF OP_EDGE0 = 4
DEF N_FIXED = 8
DEF PER_RISE = 5


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


cdef void _op_solid_ink(const Canvas* c, const Ops* t, Py_ssize_t op, Py_ssize_t by, Py_ssize_t bx,
                        uint8_t r, uint8_t g, uint8_t b, int32_t* ink, int32_t rise) noexcept nogil:
    """_op_solid for a two-pass mark: every pixel also records `rise` in ink
    (render._record_ink)."""
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
        ink[py * c.w + px] = rise


cdef void _op_restore(const Canvas* c, const Ops* t, Py_ssize_t op, Py_ssize_t by, Py_ssize_t bx,
                      const uint8_t* snap, int32_t* ink, int32_t height) noexcept nogil:
    """render._restore_under over one coverage op: ink based below `height`
    gives way to the terrain snapshot and is cleared."""
    cdef int st = _clip_state(t.ext + 4 * op, by, bx, c)
    cdef Py_ssize_t k, py, px, p
    if st == 0:
        return
    for k in range(t.start[op], t.start[op + 1]):
        py = by + t.dy[k]
        px = bx + t.dx[k]
        if st == 2 and (py < 0 or py >= c.h or px < 0 or px >= c.w):
            continue
        p = py * c.w + px
        if ink[p] < height:
            c.p[3 * p] = snap[3 * p]
            c.p[3 * p + 1] = snap[3 * p + 1]
            c.p[3 * p + 2] = snap[3 * p + 2]
            ink[p] = INK_NONE


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


cdef void _blit_ink(const Canvas* c, const uint8_t* rgba, Py_ssize_t h, Py_ssize_t w,
                    Py_ssize_t base_y, Py_ssize_t base_x, int32_t* ink, int32_t rise) noexcept nogil:
    """_blit for a two-pass sprite: every pixel with alpha > 0 also records
    `rise` in ink (render._record_ink_rgba)."""
    cdef Py_ssize_t sy0, sx0, sy1, sx1, y, x, ch
    cdef const uint8_t* q
    cdef uint8_t* d
    cdef int32_t* m
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
        m = ink + (base_y + y) * c.w + base_x + sx0
        for x in range(sx0, sx1):
            a = q[3]
            if a == 255:
                d[0] = q[0]
                d[1] = q[1]
                d[2] = q[2]
                m[0] = rise
            elif a != 0:
                for ch in range(3):
                    d[ch] = <uint8_t>((q[ch] * a + d[ch] * (255 - a)) // 255)
                m[0] = rise
            q += 4
            d += 3
            m += 1


cdef void _farm_and_grid(const Canvas* c, const Units* u, Py_ssize_t t, const Ops* edges, Py_ssize_t edge0,
                         const Stamps* g, Py_ssize_t sid, Py_ssize_t by, Py_ssize_t bx) noexcept nogil:
    """The rest of a tile's terrain half, in render._paint_terrain_*'s order:
    farm outline, then grid."""
    cdef Py_ssize_t j
    cdef uint8_t mask
    cdef const uint8_t* rgb
    if u != NULL:
        mask = u.farm_mask[t]
        if mask:
            rgb = u.farm_rgb + 3 * t
            for j in range(4):
                if mask & (1 << j):
                    _op_solid(c, edges, edge0 + j, by, bx, rgb[0], rgb[1], rgb[2])
    if sid >= 0:
        _stamp(c, g, sid, by, bx)


cdef void _unit_rows(const Canvas* c, const Units* u, Py_ssize_t t, const Ops* marks, Py_ssize_t mark_op,
                     const Ops* conform, Py_ssize_t conform_op, Py_ssize_t by, Py_ssize_t bx,
                     Py_ssize_t off_x, Py_ssize_t off_y, int32_t* ink) noexcept nogil:
    """A tile's objects half, in render._paint_objects_*'s order: marks, then
    sprites. ink, when not NULL, records each row's base rise under its pixels."""
    cdef Py_ssize_t i, r0, slot
    cdef const int32_t* row
    cdef int32_t packed
    cdef uint8_t r, g, b
    if u == NULL:
        return
    r0 = u.index[2 * t]
    for i in range(r0, r0 + u.index[2 * t + 1]):
        row = u.rows + ROW_W * i
        if row[0] == ROW_SPRITE:
            slot = row[3]
            if ink != NULL:
                _blit_ink(c, <const uint8_t*><uintptr_t>u.addrs[slot], u.dims[2 * slot], u.dims[2 * slot + 1],
                          row[1] - off_y, row[2] - off_x, ink, row[4])
            else:
                _blit(c, <const uint8_t*><uintptr_t>u.addrs[slot], u.dims[2 * slot], u.dims[2 * slot + 1],
                      row[1] - off_y, row[2] - off_x)
            continue
        packed = row[3]
        r = (packed >> 16) & 255
        g = (packed >> 8) & 255
        b = packed & 255
        if row[0] == ROW_CONFORM:
            if ink != NULL:
                _op_solid_ink(c, conform, conform_op, by, bx, r, g, b, ink, row[4])
            else:
                _op_solid(c, conform, conform_op, by, bx, r, g, b)
        elif ink != NULL:
            _op_solid_ink(c, marks, mark_op, row[1] - off_y, row[2] - off_x, r, g, b, ink, row[4])
        else:
            _op_solid(c, marks, mark_op, row[1] - off_y, row[2] - off_x, r, g, b)


cdef inline void _row_floor(const Units* u, Py_ssize_t t, int32_t* floor) noexcept nogil:
    """Lowers floor[0] to the lowest base rise among tile t's rows
    (render._ink_floor)."""
    cdef Py_ssize_t i, r0 = u.index[2 * t]
    for i in range(r0, r0 + u.index[2 * t + 1]):
        if u.rows[ROW_W * i + 4] < floor[0]:
            floor[0] = u.rows[ROW_W * i + 4]


cdef bint _two_pass_planes(const Canvas* c, uint8_t** snap, int32_t** ink) noexcept nogil:
    """Allocates pass 2's terrain snapshot (a copy of the canvas as pass 1
    left it) and its empty ink plane. False when out of memory."""
    cdef Py_ssize_t i, n = c.h * c.w
    snap[0] = <uint8_t*>malloc(max(n * 3, 1))
    ink[0] = <int32_t*>malloc(max(n, 1) * sizeof(int32_t))
    if snap[0] == NULL or ink[0] == NULL:
        return False
    memcpy(snap[0], c.p, n * 3)
    for i in range(n):
        ink[0][i] = INK_NONE
    return True


cdef void _compose(const Blend* b, Py_ssize_t spec, Py_ssize_t px, const Src* own, const Src* srcs) noexcept nogil:
    """terrain_blend._compose into b.block at the sample texels: the own
    crop, then each op. A layer is (d*(256 - m) + t*m + 128) >> 8 with m
    cropped like the texture; a neighbour (d*a + t*(128 - a) + 64) >> 7 with
    a the square-space alpha. Every value stays in 0..255, so uint8 storage
    between ops is exact, and no op reads another texel, so a subset of the
    square composes to the same bytes there."""
    # Locals, not b's fields: a uint8_t store may alias the struct, so a
    # field read in the loop would be reloaded after every store. Row
    # pointers move only when sy does (the list is row-major, for speed only).
    cdef const int32_t* samp_y = b.samp_y
    cdef const int32_t* samp_x = b.samp_x
    cdef uint8_t* block = b.block
    cdef Py_ssize_t n = b.n_samp, i, j, slot, mid, stride, ms, sy, sx, row
    cdef const uint8_t* trow = NULL
    cdef const uint8_t* mrow = NULL
    cdef const uint8_t* tex
    cdef const uint8_t* mask
    cdef const uint8_t* q
    cdef uint8_t* drow = NULL
    cdef uint8_t* d
    cdef unsigned int a, d0, d1, d2
    tex = own.tex + own.oy * own.stride + own.ox * 3
    stride = own.stride
    row = -1
    for i in range(n):
        sy = samp_y[i]
        sx = samp_x[i]
        if sy != row:
            row = sy
            trow = tex + sy * stride
            drow = block + sy * px * 3
        q = trow + sx * 3
        d = drow + sx * 3
        d0 = q[0]
        d1 = q[1]
        d2 = q[2]
        d[0] = <uint8_t>d0
        d[1] = <uint8_t>d1
        d[2] = <uint8_t>d2
    for j in range(b.start[spec], b.start[spec + 1]):
        slot = b.ops[3 * j + 1]
        mid = b.ops[3 * j + 2]
        if mid < 0 or srcs[slot].tex == NULL:
            continue
        stride = srcs[slot].stride
        ms = b.mask_stride[mid]
        tex = srcs[slot].tex + own.oy * stride + own.ox * 3
        row = -1
        if b.ops[3 * j] == BLEND_LAYER:
            mask = b.masks[mid] + own.oy * ms + own.ox
            for i in range(n):
                sy = samp_y[i]
                sx = samp_x[i]
                if sy != row:
                    row = sy
                    trow = tex + sy * stride
                    mrow = mask + sy * ms
                    drow = block + sy * px * 3
                a = mrow[sx]
                q = trow + sx * 3
                d = drow + sx * 3
                d0 = (d[0] * (256 - a) + q[0] * a + 128) >> 8
                d1 = (d[1] * (256 - a) + q[1] * a + 128) >> 8
                d2 = (d[2] * (256 - a) + q[2] * a + 128) >> 8
                d[0] = <uint8_t>d0
                d[1] = <uint8_t>d1
                d[2] = <uint8_t>d2
        else:
            mask = b.masks[mid]
            for i in range(n):
                sy = samp_y[i]
                sx = samp_x[i]
                if sy != row:
                    row = sy
                    trow = tex + sy * stride
                    mrow = mask + sy * ms
                    drow = block + sy * px * 3
                a = mrow[sx]
                # a == 128 keeps d exactly.
                if a != 128:
                    q = trow + sx * 3
                    d = drow + sx * 3
                    d0 = (d[0] * a + q[0] * (128 - a) + 64) >> 7
                    d1 = (d[1] * a + q[1] * (128 - a) + 64) >> 7
                    d2 = (d[2] * a + q[2] * (128 - a) + 64) >> 7
                    d[0] = <uint8_t>d0
                    d[1] = <uint8_t>d1
                    d[2] = <uint8_t>d2


cdef inline void _blend_src(Src* s, const Blend* b, Py_ssize_t k, Py_ssize_t px, const Src* srcs) noexcept nogil:
    """Swaps a textured candidate's crop for its composed blend block."""
    cdef Py_ssize_t spec
    if b == NULL or s.tex == NULL:
        return
    spec = b.cand[k]
    if spec < 0:
        return
    _compose(b, spec, px, s, srcs)
    s.tex = b.block
    s.stride = px * 3
    s.ox = 0
    s.oy = 0


cdef int _blend_init(Blend* b, tuple blend, Py_ssize_t n, Py_ssize_t px, list textures,
                     const int64_t[::1] tex_size) except -1:
    """Fills b from (cand, start, ops, masks, samp_y, samp_x), validating the
    spec, slot and mask indices, the mask shapes and the samples (each inside
    the px block). Crop bounds rest on the caller's tex_size and the tables'
    own checks, as for a plain crop. The caller frees b with _blend_free even
    on error."""
    if len(blend) != 6:
        raise ValueError("blend tables malformed")
    cdef const int32_t[::1] cand = blend[0]
    cdef const int64_t[::1] start = blend[1]
    cdef const int32_t[::1] ops = blend[2]
    cdef list masks = blend[3]
    cdef const int32_t[::1] samp_y = blend[4]
    cdef const int32_t[::1] samp_x = blend[5]
    cdef const uint8_t[:, ::1] mv
    cdef Py_ssize_t i, nops, nspec, nm = len(masks), nt = len(textures), size = 0, kind, slot, mid, need
    if cand.shape[0] != n or ops.shape[0] % 3 or start.shape[0] < 1:
        raise ValueError("blend tables malformed")
    if samp_y.shape[0] != samp_x.shape[0]:
        raise ValueError("blend sample columns differ in length")
    for i in range(samp_y.shape[0]):
        if not (0 <= samp_y[i] < px and 0 <= samp_x[i] < px):
            raise ValueError("blend sample outside the tile block")
    nops = ops.shape[0] // 3
    nspec = start.shape[0] - 1
    if start[0] != 0 or start[nspec] != nops:
        raise ValueError("blend spec offsets do not cover the ops")
    for i in range(nspec):
        if start[i] > start[i + 1]:
            raise ValueError("blend spec offsets decrease")
    for i in range(n):
        if cand[i] >= nspec:
            raise ValueError("candidate blend spec out of range")
    # Every crop is taken at the own texture's offset, so all must be one size.
    for i in range(nt):
        if tex_size[i]:
            if size and tex_size[i] != size:
                raise ValueError("blended textures differ in size")
            size = tex_size[i]
    shapes = [m.shape for m in masks]
    for i in range(nops):
        kind = ops[3 * i]
        slot = ops[3 * i + 1]
        mid = ops[3 * i + 2]
        if kind != BLEND_LAYER and kind != BLEND_NEIGHBOUR or not 0 <= slot < nt or mid >= nm:
            raise ValueError("blend op out of range")
        need = size if kind == BLEND_LAYER else px
        if mid >= 0 and need and shapes[mid] != (need, need):
            raise ValueError("blend mask is the wrong size")
    b.cand = _p32(cand)
    b.start = &start[0]
    b.ops = _p32(ops)
    b.masks = <const uint8_t**>malloc(max(nm, 1) * sizeof(uint8_t*))
    b.mask_stride = <Py_ssize_t*>malloc(max(nm, 1) * sizeof(Py_ssize_t))
    # Zeroed once: a texel no compose wrote reads 0 0 0, never a leftover.
    b.block = <uint8_t*>calloc(max(px * px * 3, 1), 1)
    if b.masks == NULL or b.mask_stride == NULL or b.block == NULL:
        raise MemoryError()
    b.samp_y = _p32(samp_y)
    b.samp_x = _p32(samp_x)
    b.n_samp = samp_y.shape[0]
    for i in range(nm):
        mv = masks[i]
        b.masks[i] = &mv[0, 0] if mv.shape[0] and mv.shape[1] else NULL
        b.mask_stride[i] = mv.shape[1]
    return 0


cdef void _blend_free(Blend* b) noexcept:
    free(b.masks)
    free(b.mask_stride)
    free(b.block)


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
    try:
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
    except BaseException:
        free(out)
        raise
    return out


def composite_iso(uint8_t[:, :, ::1] img, Py_ssize_t off_x, Py_ssize_t off_y,
                  const int64_t[::1] cand_x, const int64_t[::1] cand_y,
                  const int32_t[::1] cand_tex, const int32_t[::1] cand_grid,
                  const int64_t[:, :] elev, tuple geom, tuple tables, tuple textures,
                  tuple stamps, tuple units, tuple blend, bint two_pass):
    """render._paint_candidates_iso's loop over the ordered candidates.
    geom is (origin_x, origin_y, half_w, half_h, elev_step, tile_px,
    max_rise). Returns False, having painted nothing, when a neighbour rise
    is past the tables' max_rise; the caller then takes the per-tile path.
    blend is native_composite._blend_tables()' tuple plus the geometry's
    blend_samples (samp_y, samp_x), or None: a blended candidate's top block
    is composed here, at those texels only, and painted in place of its crop.
    two_pass (sprites on): every terrain half, then every tile's objects,
    with a higher nearer tile's terrain restored over lower ink first. The
    snapshot and ink plane exist only when some tile is above some ink."""
    cdef Py_ssize_t origin_x = geom[0], origin_y = geom[1], half_w = geom[2], half_h = geom[3]
    cdef Py_ssize_t step = geom[4], tile_px = geom[5], max_rise = geom[6]
    cdef Py_ssize_t n = cand_x.shape[0], mh = elev.shape[0], mw = elev.shape[1]
    cdef Py_ssize_t k, tx, ty, own, dl, dr, rul, rur, rdg, bx, by, slot, tsize, top = -INK_NONE
    cdef int32_t floor = INK_NONE
    cdef uint8_t* snap = NULL
    cdef int32_t* ink = NULL
    cdef bint seam, bad = False, oom = False
    cdef Canvas c
    cdef Ops t = _ops(tables[:9])
    cdef const uint8_t[:, ::1] luts = tables[9]
    cdef const int64_t[::1] tex_size = textures[2]
    cdef Stamps g
    cdef Units u
    cdef const Units* up = NULL
    cdef Src* srcs = NULL
    cdef Src s
    cdef Blend bl
    cdef const Blend* bp = NULL
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
    else:
        two_pass = False
    memset(&bl, 0, sizeof(Blend))
    try:
        if blend is not None:
            _blend_init(&bl, blend, n, tile_px, textures[0], tex_size)
            bp = &bl
        srcs = _sources(textures[0], textures[1])
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
                        _blend_src(&s, bp, k, tile_px, srcs)
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
                        _op_darken(&c, &t, N_FIXED + (rdg - 1) * PER_RISE + 4, by, bx)

                    _farm_and_grid(&c, up, ty * mw + tx, &t, OP_EDGE0, &g, cand_grid[k], by, bx)
                    if two_pass:
                        # The pre-scan, folded into pass 1: highest tile, lowest ink base.
                        if own * step > top:
                            top = own * step
                        _row_floor(up, ty * mw + tx, &floor)
                    else:
                        _unit_rows(&c, up, ty * mw + tx, &t, OP_DIAMOND, NULL, 0, by, bx, off_x, off_y, NULL)
                if two_pass:
                    if top > floor and not _two_pass_planes(&c, &snap, &ink):
                        oom = True
                    else:
                        for k in range(n):
                            tx = cand_x[k]
                            ty = cand_y[k]
                            own = elev[ty, tx]
                            bx = origin_x + (tx + ty) * half_w - off_x
                            by = origin_y + (ty - tx) * half_h - own * step - off_y
                            if ink != NULL and own * step > floor:
                                # render._occlude_iso: both skirts, then the diamond.
                                dl = max(own - elev[ty, tx - 1], 0) if tx > 0 else 0
                                dr = max(own - elev[ty + 1, tx], 0) if ty + 1 < mh else 0
                                if dl > 0:
                                    _op_restore(&c, &t, N_FIXED + (dl - 1) * PER_RISE + 0, by, bx, snap, ink, own * step)
                                if dr > 0:
                                    _op_restore(&c, &t, N_FIXED + (dr - 1) * PER_RISE + 1, by, bx, snap, ink, own * step)
                                _op_restore(&c, &t, OP_DIAMOND, by, bx, snap, ink, own * step)
                            _unit_rows(&c, up, ty * mw + tx, &t, OP_DIAMOND, NULL, 0, by, bx, off_x, off_y, ink)
    finally:
        free(srcs)
        _blend_free(&bl)
        free(snap)
        free(ink)
    if oom:
        raise MemoryError()
    return not bad


def composite_sloped(uint8_t[:, :, ::1] img, Py_ssize_t off_x, Py_ssize_t off_y,
                     const int64_t[::1] cand_x, const int64_t[::1] cand_y,
                     const int32_t[::1] cand_tex, const int32_t[::1] cand_grid,
                     const int32_t[::1] cand_shape, const int64_t[::1] cand_shade,
                     const int64_t[:, :] corner_rise, tuple geom, tuple quads, tuple edges,
                     tuple diamond, const float[::1] shade, tuple textures, tuple stamps, tuple units,
                     tuple blend, bint two_pass):
    """render._paint_candidates_sloped's loop. geom is (origin_x, origin_y,
    half_w, half_h, tile_px); cand_shape indexes quads (one op per shape) and
    edges (four per shape), cand_shade is each candidate's offset into
    shade. blend and two_pass as composite_iso's; a tile's occluder height
    is its lowest drawn corner and its coverage its quad."""
    cdef Py_ssize_t origin_x = geom[0], origin_y = geom[1], half_w = geom[2], half_h = geom[3]
    cdef Py_ssize_t tile_px = geom[4]
    cdef Py_ssize_t n = cand_x.shape[0], mw = corner_rise.shape[1] - 1
    cdef Py_ssize_t k, tx, ty, d_nw, d_ne, d_sw, d_se, dmin, bx, by, slot, tsize, shp, top = -INK_NONE
    cdef int32_t floor = INK_NONE
    cdef uint8_t* snap = NULL
    cdef int32_t* ink = NULL
    cdef bint oom = False
    cdef Canvas c
    cdef Ops q = _ops(quads)
    cdef Ops e = _ops(edges)
    cdef Ops dm = _ops(diamond)
    cdef const int64_t[::1] tex_size = textures[2]
    cdef Stamps g
    cdef Units u
    cdef const Units* up = NULL
    cdef Src* srcs = NULL
    cdef Src s
    cdef Blend bl
    cdef const Blend* bp = NULL
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
    else:
        two_pass = False
    memset(&bl, 0, sizeof(Blend))
    try:
        if blend is not None:
            _blend_init(&bl, blend, n, tile_px, textures[0], tex_size)
            bp = &bl
        srcs = _sources(textures[0], textures[1])
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
                    _blend_src(&s, bp, k, tile_px, srcs)
                shp = cand_shape[k]
                _op_shade(&c, &q, shp, by, bx, &s, &shade[cand_shade[k]])
                _farm_and_grid(&c, up, ty * mw + tx, &e, 4 * shp, &g, cand_grid[k], by, bx)
                if two_pass:
                    if dmin > top:
                        top = dmin
                    _row_floor(up, ty * mw + tx, &floor)
                else:
                    _unit_rows(&c, up, ty * mw + tx, &dm, 0, &q, shp, by, bx, off_x, off_y, NULL)
            if two_pass:
                if top > floor and not _two_pass_planes(&c, &snap, &ink):
                    oom = True
                else:
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
                        shp = cand_shape[k]
                        if ink != NULL and dmin > floor:
                            _op_restore(&c, &q, shp, by, bx, snap, ink, dmin)
                        _unit_rows(&c, up, ty * mw + tx, &dm, 0, &q, shp, by, bx, off_x, off_y, ink)
    finally:
        free(srcs)
        _blend_free(&bl)
        free(snap)
        free(ink)
    if oom:
        raise MemoryError()


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


# ---------------------------------------------------------------------------
# Sprites (GH #254): unit_sprites._tinted past its early return, and
# _box_resize_rgba's premultiplied area average with _area_taps' taps. float32
# op for op in numpy's order (no FMA, see tools/build_native.py), truncated.


cdef inline uint8_t _clip_u8(float v) noexcept nogil:
    if v > <float>255.0:
        return 255
    if v < <float>0.0:
        return 0
    return <uint8_t>v


def tint_rgba(uint8_t[:, :, ::1] out, const uint8_t[:, :, ::1] playercolor, int r, int g, int b):
    """unit_sprites._tinted on `out`, a copy of main, in place: each pixel
    playercolor covers (alpha > 0 and red > 0) becomes rgb * (1 - cov) +
    rgb * t * cov, with cov = red / 255 and t = (r, g, b) / 255."""
    cdef Py_ssize_t h = out.shape[0], w = out.shape[1], y, x, c
    cdef float t[3]
    cdef float cov, keep, v
    if out.shape[2] != 4 or playercolor.shape[0] != h or playercolor.shape[1] != w or playercolor.shape[2] != 4:
        raise ValueError("out and playercolor must be (h, w, 4) and the same size")
    t[0] = <float>r / <float>255.0
    t[1] = <float>g / <float>255.0
    t[2] = <float>b / <float>255.0
    with nogil:
        for y in range(h):
            for x in range(w):
                if playercolor[y, x, 3] == 0 or playercolor[y, x, 0] == 0:
                    continue
                cov = <float>playercolor[y, x, 0] / <float>255.0
                keep = <float>1.0 - cov
                for c in range(3):
                    v = <float>out[y, x, c]
                    out[y, x, c] = _clip_u8(v * keep + v * t[c] * cov)


def box_shrink_rgba(const uint8_t[:, :, ::1] rgba, const int64_t[:, ::1] ys, const float[:, ::1] wy,
                    const int64_t[:, ::1] xs, const float[:, ::1] wx, uint8_t[:, :, ::1] out):
    """unit_sprites._box_resize_rgba's shrink into out (h, w, 4), with ys/wy
    _area_taps(H, h) and xs/wx _area_taps(W, w): premultiply, the row taps
    then the column taps summed in tap order, un-premultiply where alpha > 0,
    + 0.5, clip, truncate. One output row at a time, each source row
    premultiplied once into one of two slots (taps advance a row at a time,
    so y's parity picks the slot). Exactly +0.0 is all a zero-weight row tap
    or an alpha-0 px adds, so each source row is read only across its alpha
    span, and an output px whose taps miss every span is (0, 0, 0, 0)."""
    cdef Py_ssize_t H = rgba.shape[0], W = rgba.shape[1], h = ys.shape[0], kn = ys.shape[1]
    cdef Py_ssize_t w = xs.shape[0], ln = xs.shape[1], i, j, k, c, n, x, y
    cdef Py_ssize_t x0, x1, ux0, ux1, px0 = 0, px1 = 0
    cdef float* row = NULL
    cdef float* prem = NULL
    cdef float* p
    cdef const float* r
    cdef const float* wrow
    cdef const int64_t* xrow
    cdef Py_ssize_t slot_y[2]
    cdef Py_ssize_t slot_x0[2]
    cdef Py_ssize_t slot_x1[2]
    cdef const uint8_t* s
    cdef uint8_t* o
    cdef float acc[4]
    cdef float wt, ps, a
    if rgba.shape[2] != 4 or out.shape[2] != 4:
        raise ValueError("rgba and out must have 4 channels")
    if wy.shape[0] != h or wy.shape[1] != kn or wx.shape[0] != w or wx.shape[1] != ln:
        raise ValueError("each axis needs as many weights as taps")
    if out.shape[0] != h or out.shape[1] != w:
        raise ValueError("out must be (len(ys), len(xs), 4)")
    for i in range(h):
        for k in range(kn):
            if ys[i, k] < 0 or ys[i, k] >= H:
                raise ValueError("a row tap is outside rgba")
    for j in range(w):
        for k in range(ln):
            if xs[j, k] < 0 or xs[j, k] >= W:
                raise ValueError("a column tap is outside rgba")
    if h == 0 or w == 0:
        return
    row = <float*>malloc(W * 4 * sizeof(float))
    prem = <float*>malloc(2 * W * 4 * sizeof(float))
    if row == NULL or prem == NULL:
        free(row)
        free(prem)
        raise MemoryError()
    slot_y[0] = slot_y[1] = -1
    memset(row, 0, W * 4 * sizeof(float))
    try:
        with nogil:
            for i in range(h):
                # Only the previous row's union span can be non-zero.
                memset(row + 4 * px0, 0, (px1 - px0) * 4 * sizeof(float))
                ux0 = W
                ux1 = 0
                for k in range(kn):
                    wt = wy[i, k]
                    if wt == 0:
                        continue
                    y = ys[i, k]
                    p = prem + (y & 1) * W * 4
                    if slot_y[y & 1] != y:
                        slot_y[y & 1] = y
                        s = &rgba[y, 0, 0]
                        x0 = 0
                        x1 = W
                        while x0 < W and s[4 * x0 + 3] == 0:
                            x0 += 1
                        while x1 > x0 and s[4 * x1 - 1] == 0:
                            x1 -= 1
                        slot_x0[y & 1] = x0
                        slot_x1[y & 1] = x1
                        for x in range(x0, x1):
                            n = 4 * x
                            ps = <float>s[n + 3] / <float>255.0
                            p[n] = <float>s[n] * ps
                            p[n + 1] = <float>s[n + 1] * ps
                            p[n + 2] = <float>s[n + 2] * ps
                            p[n + 3] = <float>s[n + 3]
                    x0 = slot_x0[y & 1]
                    x1 = slot_x1[y & 1]
                    if x0 >= x1:
                        continue
                    if x0 < ux0:
                        ux0 = x0
                    if x1 > ux1:
                        ux1 = x1
                    for n in range(4 * x0, 4 * x1):
                        row[n] = row[n] + wt * p[n]
                if ux0 < ux1:
                    px0 = ux0
                    px1 = ux1
                else:
                    px0 = px1 = 0
                # Every column tap, zero weights included, and the where(a > 0, a, 1) divide, as numpy does.
                o = &out[i, 0, 0]
                for j in range(w):
                    n = 4 * j
                    if xs[j, ln - 1] < px0 or xs[j, 0] >= px1:
                        memset(o + n, 0, 4)
                        continue
                    wrow = &wx[j, 0]
                    xrow = &xs[j, 0]
                    acc[0] = acc[1] = acc[2] = acc[3] = 0
                    for k in range(ln):
                        wt = wrow[k]
                        r = row + 4 * xrow[k]
                        acc[0] = acc[0] + wt * r[0]
                        acc[1] = acc[1] + wt * r[1]
                        acc[2] = acc[2] + wt * r[2]
                        acc[3] = acc[3] + wt * r[3]
                    a = acc[3] if acc[3] > 0 else <float>1.0
                    o[n] = _clip_u8(acc[0] * <float>255.0 / a + <float>0.5)
                    o[n + 1] = _clip_u8(acc[1] * <float>255.0 / a + <float>0.5)
                    o[n + 2] = _clip_u8(acc[2] * <float>255.0 / a + <float>0.5)
                    o[n + 3] = _clip_u8(acc[3] + <float>0.5)
    finally:
        free(row)
        free(prem)


def box_reduce2(const uint8_t[:, :, ::1] src, uint8_t[:, :, ::1] out):
    """render_cache._box_reduce2_numpy into out (ceil(H/2), ceil(W/2), C):
    (a + b + c + d + 2) >> 2 per channel over each 2x2 block, an odd last
    row or column paired with itself."""
    cdef Py_ssize_t H = src.shape[0], W = src.shape[1], C = src.shape[2]
    cdef Py_ssize_t h = (H + 1) // 2, w = (W + 1) // 2, i, j, c, y0, y1, x0, x1
    if out.shape[0] != h or out.shape[1] != w or out.shape[2] != C:
        raise ValueError("out must be (ceil(H/2), ceil(W/2), C) of src")
    if H == 0 or W == 0:
        return
    with nogil:
        for i in range(h):
            y0 = 2 * i
            y1 = y0 + 1 if y0 + 1 < H else y0
            for j in range(w):
                x0 = 2 * j
                x1 = x0 + 1 if x0 + 1 < W else x0
                for c in range(C):
                    out[i, j, c] = <uint8_t>((<unsigned int>src[y0, x0, c] + src[y0, x1, c]
                                              + src[y1, x0, c] + src[y1, x1, c] + 2) >> 2)
