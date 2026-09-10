"""iso_geometry.index_extent() and the scalar bounds pre-check it feeds.

Two separate bars here, and the second one is the reason this file exists.

The first is that one shared memo is legal for all nine index producers:
every one of them returns dst_y as element 0 and dst_x as element 1,
whatever trails behind (src arrays, depth/span, uv_idx). Every unpack site
in render.py reads them that way (`dst_y, dst_x, ... =
iso_geometry.<producer>(...)`), so the convention is asserted here rather
than left implicit, and the extents themselves are checked against a
brute-force min/max over the very same arrays.

The second is the pre-check inside render._clipped_paint /
render._clipped_darken. That fast path is byte-identity work, and the
repo's existing pixel oracles cannot catch a break in it: they are
self-consistency checks that run both sides of the comparison through the
same helper, so a predicate that is wrong in the same way on both sides
stays green. The equivalence tests below instead render the SAME index set
into two identical canvases, once with extent=None (today's mask path) and
once with the real extent, and compare bytes. Every boundary the predicate
can get wrong by one is covered: flush against each edge from the inside,
one pixel past each edge, and wholly clear of each edge.
"""

import numpy as np
import pytest

from descape import iso_geometry, render

TILE_PX = (8, 16, 32, 64, 128)
"""The full live tile_px set: render.SMALL/LARGE_MAP_TILE_PIXELS (64, 32)
plus the mip ladder either side of them."""

DROP_PX = (1, 2, 3, 7, 16, 40)
RISE_PX = (1, 2, 3, 8, 17, 64, 200)
"""RISE_PX deliberately runs past 2*half_h at the small tile_px values, so
shadow_quad_indices/shadow_apex_indices really do return an empty set here
(a caster fully hiding its back neighbour) rather than the None branch
going untested. test_none_iff_empty asserts that actually happened."""

CORNERS = (
    (0, 0, 0, 0),
    (5, 5, 5, 5),
    (0, 4, 0, 4),
    (8, 0, 0, 8),
    (3, 7, 1, 9),
    (12, 0, 6, 6),
)
"""Corner pixel-rise sets for the two sloped producers. The first two are
the equal-corner branch (which aliases diamond_indices /
tile_edge_indices); the rest are genuinely warped."""

EDGE_SIDES = ("left", "right", "up_left", "up_right")


def _producer_keys() -> list[tuple[object, tuple]]:
    """(producer, key) for all nine producers over a real key spread."""
    pairs: list[tuple[object, tuple]] = []
    for tile_px in TILE_PX:
        pairs.append((iso_geometry.diamond_indices, (tile_px,)))
        pairs.append((iso_geometry.seam_apex_indices, (tile_px,)))
        for side in ("left", "right"):
            for drop_px in DROP_PX:
                pairs.append((iso_geometry.skirt_quad_indices, (tile_px, drop_px, side)))
        for side in ("up_left", "up_right"):
            pairs.append((iso_geometry.seam_edge_indices, (tile_px, side)))
            for rise_px in RISE_PX:
                pairs.append((iso_geometry.shadow_quad_indices, (tile_px, rise_px, side)))
        for rise_px in RISE_PX:
            pairs.append((iso_geometry.shadow_apex_indices, (tile_px, rise_px)))
        for side in EDGE_SIDES:
            pairs.append((iso_geometry.tile_edge_indices, (tile_px, side)))
            for corners in CORNERS:
                pairs.append((iso_geometry.sloped_tile_edge_indices, (tile_px, side, *corners)))
        for corners in CORNERS:
            pairs.append((iso_geometry.sloped_quad_indices, (tile_px, *corners)))
    return pairs


def _pair_id(pair: tuple[object, tuple]) -> str:
    producer, key = pair
    return f"{producer.__name__}{key}"


PRODUCER_KEYS = _producer_keys()
PRODUCER_IDS = [_pair_id(p) for p in PRODUCER_KEYS]


@pytest.mark.parametrize("pair", PRODUCER_KEYS, ids=PRODUCER_IDS)
def test_elements_0_and_1_are_the_dst_index_arrays(pair):
    """The invariant that makes ONE shared memo legal for nine producers.

    Establishing grep, run against descape/render.py: every unpack site is
    `dst_y, dst_x, src_y, src_x = iso_geometry.diamond_indices(...)`,
    `... = iso_geometry.skirt_quad_indices(...)`, `s_dst_y, s_dst_x,
    _depth, _span = iso_geometry.shadow_quad_indices(...)`, `a_dst_y,
    a_dst_x, _depth, _span = iso_geometry.shadow_apex_indices(...)`,
    `seam_dst_y, seam_dst_x = iso_geometry.seam_edge_indices(...)`,
    `apex_dst_y, apex_dst_x = iso_geometry.seam_apex_indices(...)`,
    `dst_y, dst_x = iso_geometry.tile_edge_indices(...)`, `dst_y, dst_x,
    _src_y, _src_x, _uv_idx = iso_geometry.sloped_quad_indices(...)` and
    `edge_dst_y, edge_dst_x = iso_geometry.sloped_tile_edge_indices(...)`.
    Destination y first, destination x second, in all nine."""
    producer, key = pair
    out = producer(*key)
    dst_y, dst_x = out[0], out[1]
    assert isinstance(dst_y, np.ndarray)
    assert isinstance(dst_x, np.ndarray)
    assert dst_y.dtype.kind == "i", dst_y.dtype
    assert dst_x.dtype.kind == "i", dst_x.dtype
    assert dst_y.ndim == 1
    assert dst_x.ndim == 1
    assert dst_y.shape == dst_x.shape


@pytest.mark.parametrize("pair", PRODUCER_KEYS, ids=PRODUCER_IDS)
def test_index_extent_matches_brute_force(pair):
    """index_extent == (dy.min(), dy.max(), dx.min(), dx.max()), inclusive.

    Brute-forced off the producer's own arrays rather than off [0]/[-1]:
    none of these index sets is globally sorted, so a first/last read would
    agree with min/max only by accident."""
    producer, key = pair
    dst_y, dst_x = producer(*key)[:2]
    extent = iso_geometry.index_extent(producer, *key)
    if dst_y.size == 0:
        assert extent is None
        return
    assert extent == (int(dst_y.min()), int(dst_y.max()), int(dst_x.min()), int(dst_x.max()))
    assert all(isinstance(v, int) for v in extent)


def test_none_iff_empty():
    """None exactly when the index set is empty, and the empty branch is
    genuinely reached by this file's key spread rather than untested."""
    empties = 0
    for producer, key in PRODUCER_KEYS:
        dst_y = producer(*key)[0]
        extent = iso_geometry.index_extent(producer, *key)
        assert (extent is None) == (dst_y.size == 0), (producer.__name__, key)
        empties += dst_y.size == 0
    assert empties > 0, "no key in this file produced an empty index set"


def test_extent_is_not_read_off_the_array_ends():
    """At least one producer's dst_y really does disagree with a naive
    [0]/[-1] read, so the min/max form is load-bearing, not defensive."""
    disagreements = 0
    for producer, key in PRODUCER_KEYS:
        dst_y, dst_x = producer(*key)[:2]
        if dst_y.size == 0:
            continue
        if (int(dst_y[0]), int(dst_y[-1])) != (int(dst_y.min()), int(dst_y.max())):
            disagreements += 1
        elif (int(dst_x[0]), int(dst_x[-1])) != (int(dst_x.min()), int(dst_x.max())):
            disagreements += 1
    assert disagreements > 0


# --- the pre-check equivalence bar -------------------------------------

EQUIV_KEYS = [
    (iso_geometry.diamond_indices, (32,)),
    (iso_geometry.diamond_indices, (128,)),
    (iso_geometry.skirt_quad_indices, (32, 7, "left")),
    (iso_geometry.skirt_quad_indices, (64, 3, "right")),
    # dst_y DESCENDS within a column here, the case a [0]/[-1] extent
    # would get backwards.
    (iso_geometry.shadow_quad_indices, (64, 8, "up_left")),
    (iso_geometry.shadow_quad_indices, (32, 3, "up_right")),
    (iso_geometry.shadow_apex_indices, (64, 8)),
    (iso_geometry.seam_edge_indices, (32, "up_left")),
    (iso_geometry.seam_edge_indices, (64, "up_right")),
    (iso_geometry.seam_apex_indices, (32,)),
    (iso_geometry.tile_edge_indices, (32, "left")),
    (iso_geometry.tile_edge_indices, (64, "up_right")),
    (iso_geometry.sloped_quad_indices, (32, 0, 0, 0, 0)),
    (iso_geometry.sloped_quad_indices, (64, 3, 7, 1, 9)),
    (iso_geometry.sloped_tile_edge_indices, (32, "right", 0, 0, 0, 0)),
    (iso_geometry.sloped_tile_edge_indices, (64, "up_left", 12, 0, 6, 6)),
]
EQUIV_IDS = [_pair_id(p) for p in EQUIV_KEYS]

PAD = 6
"""Canvas margin around the index set's own extent, so the "wholly inside"
and single-edge cases below have room that is genuinely clear."""


def _offset_cases(extent: tuple[int, int, int, int], h: int, w: int) -> list[tuple[str, int, int]]:
    """(label, base_y, base_x) covering every way the predicate can be off
    by one. Labels state the intent; a degenerate single-row or
    single-column extent collapses a few "straddle" cases into wholly
    outside, which is still a case worth comparing."""
    y_lo, y_hi, x_lo, x_hi = extent
    iy, ix = -y_lo + PAD // 2, -x_lo + PAD // 2
    y_span, x_span = y_hi - y_lo + 1, x_hi - x_lo + 1
    return [
        ("inside_centre", iy, ix),
        ("inside_flush_top_left", -y_lo, -x_lo),
        ("inside_flush_bottom_right", h - 1 - y_hi, w - 1 - x_hi),
        ("straddle_top_1px", -y_lo - 1, ix),
        ("straddle_bottom_1px", h - y_hi, ix),
        ("straddle_left_1px", iy, -x_lo - 1),
        ("straddle_right_1px", iy, w - x_hi),
        ("straddle_top_half", -y_lo - y_span // 2, ix),
        ("straddle_left_half", iy, -x_lo - x_span // 2),
        # The last sliver still inside: one more pixel out and the set is
        # wholly outside. These are what pin the wholly-outside branch's
        # four comparisons, which the "outside_*" cases above sit one step
        # past and so cannot distinguish from an off-by-one.
        ("straddle_top_last_row", -y_hi, ix),
        ("straddle_bottom_last_row", h - 1 - y_lo, ix),
        ("straddle_left_last_col", iy, -x_hi),
        ("straddle_right_last_col", iy, w - 1 - x_lo),
        ("outside_top", -y_hi - 1, ix),
        ("outside_bottom", h - y_lo, ix),
        ("outside_left", iy, -x_hi - 1),
        ("outside_right", iy, w - x_lo),
        ("straddle_corner_top_left", -y_lo - 1, -x_lo - 1),
        ("straddle_corner_bottom_right", h - y_hi, w - x_hi),
        ("outside_corner_top_left", -y_hi - 1, -x_hi - 1),
        ("outside_corner_bottom_right", h - y_lo, w - x_lo),
    ]


@pytest.mark.parametrize("pair", EQUIV_KEYS, ids=EQUIV_IDS)
def test_clipped_paint_extent_matches_the_mask_path(pair):
    """_clipped_paint(extent=...) is byte-identical to extent=None."""
    producer, key = pair
    dst_y, dst_x = producer(*key)[:2]
    extent = iso_geometry.index_extent(producer, *key)
    assert extent is not None
    y_lo, y_hi, x_lo, x_hi = extent
    h, w = (y_hi - y_lo + 1) + PAD, (x_hi - x_lo + 1) + PAD
    rng = np.random.default_rng(0xC0FFEE)
    base = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    values = rng.integers(0, 256, size=(dst_y.size, 3), dtype=np.uint8)
    for label, base_y, base_x in _offset_cases(extent, h, w):
        masked, fast = base.copy(), base.copy()
        render._clipped_paint(masked, base_y, base_x, dst_y, dst_x, values)
        render._clipped_paint(fast, base_y, base_x, dst_y, dst_x, values, extent=extent)
        assert np.array_equal(masked, fast), label


@pytest.mark.parametrize("pair", EQUIV_KEYS, ids=EQUIV_IDS)
def test_clipped_paint_extent_matches_on_an_id_plane(pair):
    """The same bar on a 2-D int32 canvas: _render_tile_sloped_ids paints
    the pick plane through this helper, not a 3-channel image."""
    producer, key = pair
    dst_y, dst_x = producer(*key)[:2]
    extent = iso_geometry.index_extent(producer, *key)
    assert extent is not None
    y_lo, y_hi, x_lo, x_hi = extent
    h, w = (y_hi - y_lo + 1) + PAD, (x_hi - x_lo + 1) + PAD
    rng = np.random.default_rng(0x1D)
    base = rng.integers(-2, 4096, size=(h, w), dtype=np.int32)
    values = np.full(dst_y.size, 7, dtype=np.int32)
    for label, base_y, base_x in _offset_cases(extent, h, w):
        masked, fast = base.copy(), base.copy()
        render._clipped_paint(masked, base_y, base_x, dst_y, dst_x, values)
        render._clipped_paint(fast, base_y, base_x, dst_y, dst_x, values, extent=extent)
        assert np.array_equal(masked, fast), label


@pytest.mark.parametrize("pair", EQUIV_KEYS, ids=EQUIV_IDS)
def test_clipped_darken_extent_matches_the_mask_path(pair):
    """_clipped_darken(extent=...) is byte-identical to extent=None.

    Factors are a real float32 spread, never all-1.0: at 1.0 the
    multiply-and-truncate is an exact round trip, so an all-1.0 control
    would pass even if the fast path scattered the wrong pixels."""
    producer, key = pair
    dst_y, dst_x = producer(*key)[:2]
    extent = iso_geometry.index_extent(producer, *key)
    assert extent is not None
    y_lo, y_hi, x_lo, x_hi = extent
    h, w = (y_hi - y_lo + 1) + PAD, (x_hi - x_lo + 1) + PAD
    rng = np.random.default_rng(0xDA24E4)
    base = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    factors = rng.uniform(0.3, 1.0, size=dst_y.size).astype(np.float32)
    for label, base_y, base_x in _offset_cases(extent, h, w):
        masked, fast = base.copy(), base.copy()
        render._clipped_darken(masked, base_y, base_x, dst_y, dst_x, factors.copy())
        render._clipped_darken(fast, base_y, base_x, dst_y, dst_x, factors.copy(), extent=extent)
        assert np.array_equal(masked, fast), label


def test_extent_none_keeps_todays_path_on_an_empty_set():
    """An empty index set memoizes to None, and None must fall straight
    through to the existing inert mask path rather than into either
    branch of the pre-check."""
    extent = iso_geometry.index_extent(iso_geometry.shadow_quad_indices, 8, 16, "up_left")
    assert extent is None
    dst_y, dst_x = iso_geometry.shadow_quad_indices(8, 16, "up_left")[:2]
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    before = img.copy()
    render._clipped_darken(img, 0, 0, dst_y, dst_x, np.zeros(0, dtype=np.float32), extent=extent)
    assert np.array_equal(img, before)
