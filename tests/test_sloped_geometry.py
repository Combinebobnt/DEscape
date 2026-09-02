"""Verifies descape.iso_geometry's Phase 6 (Sloped) additions --
corner_rise_px and sloped_quad_indices -- the load-bearing partition proof
for the sloped warp, written before render.py's compositor was built on top
of it.

Checks:
  1. Equal-corner delegation: sloped_quad_indices returns diamond_indices'
     own arrays verbatim (identity, not merely equal) when all four corners
     are equal, plus _identity_uv_idx as its fifth -- what lets
     render_terrain_sloped's flat-map oracle against render_terrain_iso be
     byte-identical rather than approximate.
  2. corner_rise_px's "average" rule is exact on a flat map: every corner
     equals e * elev_step, not an off-by-rounding approximation, for every
     touching-tile count from 1 (map corner) to 4 (interior vertex).
  3. corner_rise_px's max/min rules pick the real max/min of touching tile
     elevations, scaled by elev_step, at a deliberately non-flat corner.
  3b. The rule render.py actually ships is the one the game was measured to
     use. Nothing else here depends on WHICH rule ships, so without this
     the constant could be changed back and no test would notice.
  3c. On real corpus terrain, at every elev_step_pct stop: no tile paints
     off-canvas, and no column it paints has a hole (corpus tier).
  4. Partition, realistic slope magnitude: for a synthetic map carrying a
     gentle multi-tile ramp and a steep single-tile ramp (both within this
     project's own ±1-elevation-delta-between-neighbors invariant, see
     iso_geometry.ELEV_STEP_DIVISOR's own comment), the union of every
     tile's sloped_quad_indices destination pixels has zero overlap AND no
     unpainted interior row -- the second half is the actual seam gate, and
     the reason the check is canvas-backed rather than dict-backed.
  5. Per-column coverage, single tile: every screen column a tile paints is
     a CONTIGUOUS run of destination rows with no repeats, across a broad
     set of corner shapes check 4 never reaches.
  6. Shared-edge abutment between tiles (test_adjacent_tiles_abut_exactly)
     -- what replaced the old rigid-translation argument once columns
     became variable-length, and the check that actually rules out the seam
     lattice rather than merely the within-tile arcs.
  7. uv_idx alignment: every output pixel resamples a diamond pixel in its
     OWN column, so render._slope_shade gathered through it stays 1:1 with
     the resampled output.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import iso_geometry as ig


def test_sloped_quad_indices_delegates_at_equal_corners():
    """Asserted by object IDENTITY, not equality: the flat path's whole
    point is that no independently-derived arithmetic runs, so a copy that
    merely compares equal today would still be free to drift by a ULP at a
    floor() boundary tomorrow and break the byte-identity oracle."""
    tile_px = 32
    base = ig.diamond_indices(tile_px)
    for value in (0, 5, -3, 17):
        got = ig.sloped_quad_indices(tile_px, value, value, value, value)
        assert len(got) == 5, "the sloped contract is a 5-tuple, flat path included"
        for a, b in zip(got[:4], base):
            assert a is b, "the flat path must hand back diamond_indices' own arrays"
        assert got[4] is ig._identity_uv_idx(tile_px), "the fifth must be the cached no-op gather"


@pytest.mark.parametrize("touching", [1, 2, 3, 4])
def test_corner_rise_average_exact_on_flat_map(touching):
    # A flat map (uniform elevation e) must yield corner_rise == e *
    # elev_step at EVERY corner regardless of how many tiles touch it
    # (map corner=1, edge=2, interior=4) -- corner_rise_px's own docstring
    # names this as the property the "sum first, divide once" formula
    # exists to guarantee.
    e = 3
    w = h = 4
    elevations = np.full((h, w), e, dtype=np.int64)
    proj = ig.canvas_size_and_origin(w, h, 32, 0, e)
    rise = ig.corner_rise_px(elevations, proj, rule="average")
    assert rise.shape == (h + 1, w + 1)
    assert np.all(rise == e * proj.elev_step)
    # Sanity on the `touching` parametrization itself: confirm each named
    # corner really does have that many touching tiles for this map size.
    counts = {
        1: [(0, 0), (0, w), (h, 0), (h, w)],
        2: [(0, 1), (1, 0)],
        3: [],  # a plain rectangular map has no 3-touching vertex
        4: [(1, 1), (2, 2)],
    }
    for cy, cx in counts[touching]:
        n = sum(1 for dy, dx in ((-1, -1), (-1, 0), (0, -1), (0, 0)) if 0 <= cy + dy < h and 0 <= cx + dx < w)
        assert n == touching


def test_corner_rise_max_min_rules():
    # 2x2 map, elevations chosen so the shared interior corner (1, 1) sees
    # all four distinct values -- max/min must pick the real extremes.
    elevations = np.array([[0, 2], [5, 1]], dtype=np.int64)
    proj = ig.canvas_size_and_origin(2, 2, 32, 0, 5)
    rise_max = ig.corner_rise_px(elevations, proj, rule="max")
    rise_min = ig.corner_rise_px(elevations, proj, rule="min")
    assert rise_max[1, 1] == 5 * proj.elev_step
    assert rise_min[1, 1] == 0 * proj.elev_step
    # A map-corner vertex (only one touching tile) is unambiguous under
    # every rule.
    assert rise_max[0, 0] == rise_min[0, 0] == elevations[0, 0] * proj.elev_step


def test_shipped_corner_rule_matches_the_in_game_captures():
    """Pins render.SLOPE_CORNER_RULE against what DE was measured to do.

    Nothing else in this suite depends on WHICH rule ships -- the partition,
    abutment and clamp checks all hold under any of the three, and the
    flat-map byte-identity oracle is rule-invariant by construction. So
    flipping the constant used to move no test at all, which is exactly how
    a measured answer gets quietly undone later.

    The shape below is `lone_pit` from tools/gen_elevation_reference.py: one
    tile a level lower inside a raised block. In the 2026-08-22 in-game
    captures DE draws that with a perfectly regular tile grid over the pit
    -- no dimple. "min" would sink those corners a whole level and
    "average" a quarter of one; only "max" predicts the null that was
    actually observed.
    """
    from descape.render import SLOPE_CORNER_RULE

    assert SLOPE_CORNER_RULE == "max"

    elevations = np.ones((5, 5), dtype=np.int64)
    elevations[2, 2] = 0
    proj = ig.canvas_size_and_origin(5, 5, 32, 0, 1)
    rise = ig.corner_rise_px(elevations, proj, rule=SLOPE_CORNER_RULE)
    # The pit's own four corners, and every corner of the block around it.
    assert (rise[1:4, 1:4] == proj.elev_step).all()


def test_corner_rise_rejects_bad_rule():
    elevations = np.zeros((2, 2), dtype=np.int64)
    proj = ig.canvas_size_and_origin(2, 2, 32, 0, 0)
    with pytest.raises(ValueError):
        ig.corner_rise_px(elevations, proj, rule="mean")


def _partition_check(elevations: np.ndarray, tile_px: int, rule: str) -> tuple[int, int, int, int]:
    """(overlaps, unique, interior_gaps, emitted) for rendering every tile
    of `elevations` through sloped_quad_indices at elevation-0 baseline
    placement -- the same construction render.py's own _render_tile_sloped
    uses (tile_screen_origin(x, y, 0, proj), base_y -= d_min, paint at
    dst_y/dst_x), just without a real image or texture.

    CANVAS-backed, not dict-backed, and that is the whole upgrade: a dict
    of claimed pixels can see an overlap but is structurally blind to an
    unwritten row, which is why the predecessor of this helper passed
    happily for months while the seam lattice was on screen. A canvas can
    be asked what it did NOT paint.

    interior_gaps counts, per canvas column, unpainted pixels lying between
    that column's own topmost and bottommost painted pixel -- the black
    lattice exactly as it appears on screen. Background outside the map's
    diamond silhouette falls outside every column's [top, bottom] span by
    construction, so this needs no outline model to exclude it."""
    h, w = elevations.shape
    proj = ig.canvas_size_and_origin(w, h, tile_px, 0, int(elevations.max()), corner_headroom_steps=1)
    corner_rise = ig.corner_rise_px(elevations, proj, rule=rule)
    count = np.zeros((proj.canvas_h, proj.canvas_w), dtype=np.int32)
    emitted = 0
    for y in range(h):
        for x in range(w):
            d_nw, d_ne, d_sw, d_se = (
                int(corner_rise[y, x]),
                int(corner_rise[y, x + 1]),
                int(corner_rise[y + 1, x]),
                int(corner_rise[y + 1, x + 1]),
            )
            d_min = min(d_nw, d_ne, d_sw, d_se)
            dst_y, dst_x, _src_y, _src_x, _uv_idx = ig.sloped_quad_indices(tile_px, d_nw, d_ne, d_sw, d_se)
            base_x, base_y = ig.tile_screen_origin(x, y, 0, proj)
            base_y -= d_min
            ay = base_y + dst_y
            ax = base_x + dst_x
            inside = (ay >= 0) & (ay < proj.canvas_h) & (ax >= 0) & (ax < proj.canvas_w)
            assert inside.all(), f"tile ({x},{y}) painted outside the canvas"
            # np.add.at, not `count[ay, ax] += 1`: buffered fancy indexing
            # would collapse a repeated pixel into one increment, which is
            # the very thing an overlap count exists to see.
            np.add.at(count, (ay, ax), 1)
            emitted += int(dst_y.size)
    painted = count > 0
    interior_gaps = 0
    for cx in range(proj.canvas_w):
        rows = np.flatnonzero(painted[:, cx])
        if rows.size:
            interior_gaps += int((~painted[rows[0] : rows[-1] + 1, cx]).sum())
    return int((count > 1).sum()), int(painted.sum()), interior_gaps, emitted


def _assert_perfect_partition(elevations: np.ndarray, tile_px: int) -> None:
    overlaps, unique, interior_gaps, emitted = _partition_check(elevations, tile_px, rule="average")
    assert overlaps == 0, f"{overlaps} pixel(s) claimed by more than one tile"
    assert interior_gaps == 0, f"{interior_gaps} unpainted interior pixel(s) -- the seam lattice"
    assert unique == emitted, "a perfect partition paints exactly as many pixels as it emits"
    # Non-vacuity. Fixed-length columns would satisfy all three assertions
    # above while restoring the old geometry wholesale, so pin that the
    # resample really is variable-length: a rigid translation always emits
    # exactly one diamond's worth of pixels per tile.
    assert unique != elevations.size * ig.diamond_indices(tile_px)[0].size, (
        "column lengths did not vary -- this looks like a revert to the rigid per-column shift"
    )


def test_partition_gentle_multi_tile_ramp():
    tile_px = 16
    w = h = 6
    elevations = np.zeros((h, w), dtype=np.int64)
    elevations[:, w // 2 :] = 1  # climbs by one full elevation level over the map's width
    _assert_perfect_partition(elevations, tile_px)


def test_partition_steep_single_tile_ramp():
    # The same one-level rise concentrated into a single tile-width seam --
    # the steepest ramp this project's own ±1 real elevation invariant ever
    # produces (docs/PLAN_V2_6.md's Track C: "steep_seam" tests DELTA>=2,
    # which is out of scope for a ramp -- MapManager never creates one).
    tile_px = 16
    w = h = 6
    elevations = np.zeros((h, w), dtype=np.int64)
    elevations[:, w // 2 + 1 :] = 1
    _assert_perfect_partition(elevations, tile_px)


def _column_shapes(tile_px: int, corners: tuple[int, int, int, int]) -> tuple[int, int]:
    """(gap_rows, duplicate_rows) summed over every screen column of one
    tile's sloped_quad_indices output -- a gap row is a destination row
    inside a column's own [min, max] span that no source pixel writes, a
    duplicate is two source pixels landing on the same (column, row)."""
    dst_y, dst_x, _src_y, _src_x, _uv_idx = ig.sloped_quad_indices(tile_px, *corners)
    gaps = duplicates = 0
    for col in np.unique(dst_x):
        rows = np.sort(dst_y[dst_x == col])
        span = int(rows[-1] - rows[0]) + 1
        distinct = int(np.unique(rows).size)
        gaps += span - distinct
        duplicates += int(rows.size) - distinct
    return gaps, duplicates


# Deliberately wider than the (0, 1, 0, 1)-shaped ramps checks 1-4 reach:
# single-corner bumps, edge ramps in all 4 orientations, both diagonals,
# and a three-corner combo -- scaled by three deltas, then two asymmetric
# mixed shapes no uniform scaling produces.
_UNIT_SHAPES = [
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 1, 0, 0),
    (0, 0, 1, 1),
    (1, 0, 1, 0),
    (0, 1, 0, 1),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (1, 1, 1, 0),
]
# delta 8 is not a synthetic extreme: elev_step == half_h //
# ELEV_STEP_DIVISOR == tile_px // 8, so a single elevation level is
# exactly 8px of corner rise at tile_px=64.
_CORNER_SHAPES = [tuple(v * d for v in shape) for d in (1, 2, 8) for shape in _UNIT_SHAPES] + [
    (1, 2, 0, 3),
    (3, 1, 2, 0),
]


@pytest.mark.parametrize("tile_px", [ig.MIP_MIN_TILE_PIXELS, 16, 64, ig.MIP_MAX_TILE_PIXELS])
@pytest.mark.parametrize("corners", _CORNER_SHAPES)
def test_sloped_quad_indices_columns_stay_contiguous(tile_px, corners):
    """Every screen column of a sloped tile must be a contiguous run of
    destination rows, with no row written twice.

    Regression for a real defect: sloped_quad_indices used to round each
    pixel's bilinear `rise` INDEPENDENTLY. Within one column, `rise` varies
    over its full corner-to-corner range (the centre column traverses the
    NE corner value to the SW one), so wherever `rise` decreased down a
    column, round() ticked down by 1 across some row -- and that -1
    cancelled the +1 step dst_y already takes, so two source pixels
    collapsed onto one destination row and the row between them was never
    written. Against a zeroed canvas that left a thin black seam tracking
    the bilinear iso-contour: on screen, nested dark arcs inside every
    sloped tile.

    Checks 1-4 above never caught it because both of their configs reduce
    to a (0, 1, 0, 1)-shaped ramp.

    THIS TEST IS NO LONGER THE THING STANDING BETWEEN THE CODE AND THE
    SEAMS, and its old docstring's argument for why is now void. That
    argument ran: fixed pixel count + contiguity + no duplicates implies a
    rigid translation, which cannot open a gap. Columns are variable-length
    now, so the fixed pixel count is gone and the implication with it. What
    rules out seams instead is two independent properties, neither of them
    here: shared-edge agreement (test_adjacent_tiles_abut_exactly) and the
    max(raw_len, 1) clamp only ever LENGTHENING a run, so a hole is
    unreachable for any corner input at all.

    What survives, and why it is still worth running: contiguity and
    no-duplicates are exactly the WITHIN-tile arcs defect above, which is a
    different bug from the between-tile lattice and is not covered by
    either replacement. All 140 parametrizations stay for that reason."""
    gaps, duplicates = _column_shapes(tile_px, corners)
    assert gaps == 0, f"{gaps} unwritten row(s) inside a column at tile_px={tile_px}, corners={corners}"
    assert duplicates == 0, f"{duplicates} doubly-written row(s) at tile_px={tile_px}, corners={corners}"


def test_diamond_column_runs_slices_match_diamond_indices():
    """_diamond_column_runs' (col_starts, lens, order) triple must actually
    slice diamond_indices' pixels into per-column top-to-bottom runs.

    This is the arithmetic the whole resample gathers through, and an
    off-by-one in it mis-reads texels silently -- no crash, no coverage
    change, just wrong colours. Checked here, independently of the sloped
    path, because the alternative is only finding out through a render.

    The specific trap it pins is the indexing convention: col_starts is
    indexed by ABSOLUTE column, not by position within `cols`. Those two
    differ by exactly the one unused leading column, so confusing them
    shifts every gather by one column's worth of pixels."""
    for tile_px in (8, 16, 64, 128):
        cols, tops, lens, col_starts, order = ig._diamond_column_runs(tile_px)
        dst_y, dst_x, _src_y, _src_x = ig.diamond_indices(tile_px)
        half_w, _half_h = ig.half_dims(tile_px)
        assert order.size == dst_y.size
        assert int(lens.sum()) == dst_y.size, "columns must tile the pixel set exactly"
        for c in cols.tolist():
            sl = order[col_starts[c] : col_starts[c] + lens[c]]
            assert (dst_x[sl] == c).all(), f"column {c} slice leaks into another column"
            rows = dst_y[sl]
            assert rows[0] == tops[c], f"column {c} slice does not start at its top row"
            assert np.all(np.diff(rows) == 1), f"column {c} slice is not top-to-bottom contiguous"
        # The two edge columns hold no pixels and must stay empty rather
        # than silently borrowing a neighbour's run.
        assert lens[0] == 0 and lens[2 * half_w - 1] == 0


@pytest.mark.parametrize("tile_px", [8, 16, 64, 128])
def test_identity_uv_idx_is_a_cached_readonly_identity(tile_px):
    """_identity_uv_idx is what sloped_quad_indices' equal-corner branch
    will hand back as its fifth element, so all three of its properties are
    load-bearing rather than incidental.

    Identity: gathering `shade` through it must be a no-op, which is what
    keeps the flat path byte-identical to Stepped rather than merely close.
    Same-object caching: the equal-corner branch returns cached arrays by
    IDENTITY, and a fresh allocation per call would quietly undo that.
    Read-only: it is module-global shared state, unlike the fresh arrays
    the sloped path builds, so a caller mutating it in place would corrupt
    every later tile -- the asymmetry between the two branches is the trap
    here, and it is why the flag is set rather than assumed."""
    idx = ig._identity_uv_idx(tile_px)
    n = ig.diamond_indices(tile_px)[0].size
    assert idx.size == n
    assert np.array_equal(idx, np.arange(n))
    assert idx is ig._identity_uv_idx(tile_px), "must be cached by identity, not rebuilt"
    assert not idx.flags.writeable, "module-global shared array must not be writable"
    # The property that actually matters downstream: gathering through it
    # returns the source unchanged.
    shade = np.linspace(0.5, 1.5, n)
    assert np.array_equal(shade[idx], shade)
    with pytest.raises(ValueError):
        idx[0] = 123


@pytest.mark.parametrize("tile_px", [8, 16, 64, 128])
@pytest.mark.parametrize("corners", [(0, 8, 0, 8), (8, 0, 0, 0), (1, 2, 0, 3), (0, 0, 4, 8)])
def test_uv_idx_stays_inside_its_own_column(tile_px, corners):
    """The structural half of the _slope_shade alignment bar (the render
    half is tests/test_sloped_render.py's own).

    Today this alignment is asserted nowhere, and that is a real hole
    rather than a theoretical one: on a genuinely sloped map a mis-gathered
    shade produces no length error, no coverage change and no exception --
    only wrong pixel VALUES, which nothing off-engine looks at.

    dst_x == diamond_dst_x[uv_idx] is the load-bearing line. It says every
    output pixel resamples a diamond pixel from its OWN screen column, so
    the gather can never smear shading sideways across the tile; the
    src_y/src_x equalities alone would pass even under a column shift,
    since they are gathered through uv_idx by the same code."""
    dst_y, dst_x, src_y, src_x, uv_idx = ig.sloped_quad_indices(tile_px, *corners)
    d_dst_y, d_dst_x, d_src_y, d_src_x = ig.diamond_indices(tile_px)
    assert uv_idx.shape == dst_y.shape
    assert int(uv_idx.min()) >= 0 and int(uv_idx.max()) < d_dst_y.size
    assert np.array_equal(src_y, d_src_y[uv_idx])
    assert np.array_equal(src_x, d_src_x[uv_idx])
    assert np.array_equal(d_dst_x[uv_idx], dst_x), "a pixel resampled a texel from another column"
    # Monotone within a column: the resample walks the source run top to
    # bottom, so a slope can stretch or squash a column but never flip it.
    src_rows = d_dst_y[uv_idx]
    for c in np.unique(dst_x):
        col = src_rows[dst_x == c]
        assert np.all(np.diff(col) >= 0), f"column {c} resamples out of order"


def _patch_intervals(tile_px: int, corner_rise: np.ndarray, proj) -> dict:
    """{(x, y): {canvas_column: (first_row, last_row)}} for every tile of a
    patch, placed exactly as _render_tile_sloped places them."""
    h, w = corner_rise.shape[0] - 1, corner_rise.shape[1] - 1
    out = {}
    for y in range(h):
        for x in range(w):
            d_nw, d_ne, d_sw, d_se = (
                int(corner_rise[y, x]),
                int(corner_rise[y, x + 1]),
                int(corner_rise[y + 1, x]),
                int(corner_rise[y + 1, x + 1]),
            )
            d_min = min(d_nw, d_ne, d_sw, d_se)
            cols, run_start, run_len, _raw = ig._sloped_column_runs(
                tile_px, d_nw - d_min, d_ne - d_min, d_sw - d_min, d_se - d_min
            )
            base_x, base_y = ig.tile_screen_origin(x, y, 0, proj)
            base_y -= d_min
            out[(x, y)] = {
                base_x + int(c): (
                    base_y + int(run_start[c]),
                    base_y + int(run_start[c]) + int(run_len[c]) - 1,
                )
                for c in cols.tolist()
            }
    return out


def _abuts(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Two row intervals are disjoint and touching -- neither overlapping
    nor leaving a row between them. Stated without reference to which one
    is on top, so the test does not have to re-derive the screen ordering
    of each neighbour direction."""
    return a[1] + 1 == b[0] or b[1] + 1 == a[0]


def _patch_corner_rise(w: int, h: int, tile_px: int, pct: int, seed: int = 11):
    """A seeded elevation patch that is LEGAL terrain, plus its projection.

    The +-1-delta-between-neighbours restriction is not decoration here.
    A 2-level jump makes a one-tile slope steeper than the tile is tall at
    pct=100, which trips _sloped_column_runs' max(_, 1) clamp -- and the
    clamp lengthens a run, producing deliberate overlap. Abutment would
    then fail for a documented reason having nothing to do with the shared
    -edge cut this test exists to check. An earlier draft of this fixture
    drew from {0, 1, 2} freely and did exactly that at pct=100.

    So the legality is asserted rather than left to the seed: a future
    change to the generator cannot silently reintroduce illegal terrain
    and be read as a kernel regression."""
    rng = np.random.default_rng(seed)
    elevations = rng.integers(0, 2, size=(h, w)).astype(np.int64)
    assert np.abs(np.diff(elevations, axis=0)).max() <= 1
    assert np.abs(np.diff(elevations, axis=1)).max() <= 1
    proj = ig.canvas_size_and_origin(
        w, h, tile_px, 0, ig.MAX_ELEVATION, elev_step_pct=pct, corner_headroom_steps=1
    )
    return ig.corner_rise_px(elevations, proj, rule="average"), proj


@pytest.mark.parametrize("tile_px", [8, 16, 64, 128])
@pytest.mark.parametrize("pct", [50, 100])
def test_adjacent_tiles_abut_exactly(tile_px, pct):
    """THE replacement for the rigid-translation proof, and the check the
    old contract had no equivalent of at all.

    Every pair of edge-neighbouring tiles must abut exactly at every canvas
    column they share: no row claimed twice (overlap) and no row left
    unclaimed between them (the black seam lattice this whole change
    exists to remove). It holds because both tiles derive the shared cut
    from the same (k, A, B, den) by integer-only arithmetic.

    WRITE THE TRAP DOWN, because anyone re-deriving this test will hit it:
    demanding the same of DIAGONAL neighbours at every shared column fails
    ~46% of the time in a correct build as well as a broken one, because at
    a non-apex column another tile sits between the two. The diagonal
    (x+1, y-1) touches T at EXACTLY the two apex columns and nowhere else,
    which is what the second half of this test pins."""
    w = h = 3
    corner_rise, proj = _patch_corner_rise(w, h, tile_px, pct)
    tiles = _patch_intervals(tile_px, corner_rise, proj)
    half_w, _half_h = ig.half_dims(tile_px)

    edge_checks = 0
    for (x, y), cols_a in tiles.items():
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cols_b = tiles.get((x + dx, y + dy))
            if cols_b is None:
                continue
            for cx, span_a in cols_a.items():
                if cx not in cols_b:
                    continue
                assert _abuts(span_a, cols_b[cx]), (
                    f"tile ({x},{y}) and ({x + dx},{y + dy}) do not abut at canvas column {cx}: "
                    f"{span_a} vs {cols_b[cx]} (tile_px={tile_px}, pct={pct})"
                )
                edge_checks += 1
    assert edge_checks > 0, "the patch produced no shared edge columns to check"

    # Diagonal (x+1, y-1) shares a base_x with T, so it overlaps in every
    # column -- but geometrically it only touches at the two apex columns.
    apex_checks = 0
    for (x, y), cols_a in tiles.items():
        cols_b = tiles.get((x + 1, y - 1))
        if cols_b is None:
            continue
        base_x, _base_y = ig.tile_screen_origin(x, y, 0, proj)
        for c in (half_w - 1, half_w):
            cx = base_x + c
            if cx in cols_a and cx in cols_b:
                assert _abuts(cols_a[cx], cols_b[cx]), (
                    f"diagonal neighbour does not meet ({x},{y}) at apex column {c}"
                )
                apex_checks += 1
    assert apex_checks > 0, "the patch produced no diagonal apex columns to check"


@pytest.mark.parametrize("tile_px", [8, 16, 64, 128])
def test_apex_columns_land_exactly_on_the_shared_corner(tile_px):
    """The two apex columns must evaluate their upper cut exactly ON the NE
    corner value, and their lower cut exactly on SW -- no rounding residue.

    That exactness is what makes a tile agree with its DIAGONAL neighbour
    by construction rather than by luck, and it is why the fraction is
    k/(half_w - 1) rather than the geometrically exact (2k+1)/(2*half_w).

    Asserted directly rather than via a mutation, deliberately: the
    (2k+1)/(2*half_w) alternative was measured NOT to break abutment at any
    tested tile_px/pct, so this is structural insurance, not a load-bearing
    behavioural difference on legal maps. Claiming otherwise would overstate
    it."""
    half_w, _half_h = ig.half_dims(tile_px)
    _cols, tops, lens, _cs, _order = ig._diamond_column_runs(tile_px)
    for nw, ne, sw, se in ((0, 7, 3, 5), (5, 0, 9, 2), (0, 11, 11, 0), (4, 4, 0, 8)):
        _c, run_start, _run_len, raw_len = ig._sloped_column_runs(tile_px, nw, ne, sw, se)
        for c in (half_w - 1, half_w):
            # run_start = tops[c] - r_up, so r_up is recoverable exactly.
            r_up = int(tops[c]) - int(run_start[c])
            assert r_up == ne, f"apex column {c} upper cut is {r_up}, not the shared NE corner {ne}"
            # raw_len, NOT run_len: the max(_, 1) clamp destroys this
            # recovery wherever it binds (these deliberately steep probe
            # corners do bind it at small tile_px), which is exactly why
            # the kernel returns the pre-clamp length alongside it.
            r_lo = int(tops[c]) + int(lens[c]) - (int(run_start[c]) + int(raw_len[c]))
            assert r_lo == sw, f"apex column {c} lower cut is {r_lo}, not the shared SW corner {sw}"


@pytest.mark.parametrize("tile_px", [8, 16, 64, 128])
def test_run_rows_stay_within_derived_bounds(tile_px):
    """Replaces sloped_quad_indices' docstring claim that "dst_y's own
    smallest value stays 0", which is measured false both today and under
    the new contract.

    The real bound: 0 <= r_up, r_lo <= max(corners), so a run's rows lie in
    [-max(corners), 2*half_h - 1]. What keeps that safe on canvas is
    _clipped_paint's masking plus corner_headroom_px' sizing, NOT a
    min-0 property -- so the bound is what gets asserted, not the myth."""
    half_w, half_h = ig.half_dims(tile_px)
    for corners in ((0, 8, 0, 8), (8, 0, 0, 0), (0, 0, 0, 12), (3, 9, 1, 7)):
        cols, run_start, run_len, _raw = ig._sloped_column_runs(tile_px, *corners)
        lo = int(run_start[cols].min())
        hi = int((run_start[cols] + run_len[cols] - 1).max())
        assert lo >= -max(corners), f"run start {lo} below the derived bound at corners={corners}"
        assert hi <= 2 * half_h - 1, f"run end {hi} above the derived bound at corners={corners}"
        assert half_w >= 2  # the cut denominator half_w - 1 is never 0


@pytest.mark.parametrize("tile_px", [8, 16, 64, 128])
def test_column_length_clamp_is_unreachable_on_legal_slopes(tile_px):
    """The max(raw_len, 1) clamp must never bind at a shipped elev_step_pct
    stop below 200, and must genuinely bind at 200 -- so the branch is
    covered rather than dead, and so a future change that starts clamping
    on ordinary terrain is caught.

    pct=200 is the degenerate case where one elevation level is a 45-degree
    slope on screen. Benign (the clamp only ever lengthens a run, so it can
    cause overlap but never a hole), but it must be observable.

    elev_step is derived from the projection rather than hardcoded: the
    kernel takes pixel rises, not a pct, so hardcoding would silently
    decouple this test from the stop it claims to exercise."""
    rng = np.random.default_rng(7)
    elevations = rng.integers(0, 2, size=(9, 9)).astype(np.int64)

    def clamped_columns(pct: int) -> int:
        proj = ig.canvas_size_and_origin(
            9, 9, tile_px, 0, ig.MAX_ELEVATION, elev_step_pct=pct, corner_headroom_steps=1
        )
        corner_rise = ig.corner_rise_px(elevations, proj, rule="average")
        total = 0
        for y in range(9):
            for x in range(9):
                d = (
                    int(corner_rise[y, x]),
                    int(corner_rise[y, x + 1]),
                    int(corner_rise[y + 1, x]),
                    int(corner_rise[y + 1, x + 1]),
                )
                d_min = min(d)
                cols, _rs, _rl, raw_len = ig._sloped_column_runs(tile_px, *[v - d_min for v in d])
                total += int((raw_len[cols] < 1).sum())
        return total

    for pct in (25, 50, 100):
        assert clamped_columns(pct) == 0, f"clamp bound at a legal slope, pct={pct}, tile_px={tile_px}"
    assert clamped_columns(200) > 0, f"clamp never binds even at pct=200, tile_px={tile_px} -- dead branch"


@pytest.mark.corpus
def test_sloped_quad_indices_key_cardinality(corpus_files):
    """Measures how many distinct (normalized d_nw, d_ne, d_sw, d_se) keys
    real maps actually produce -- sloped_quad_indices' own docstring says
    its maxsize should be revisited against this measurement rather than
    assumed; this is that measurement, kept as a permanent regression
    signal rather than a one-off note. A cache miss is a perf regression,
    never a correctness one, so this only warns (via a generous upper
    bound assertion), it doesn't require an exact number."""
    from descape.render import elevations_and_proj
    from descape.scenario_io import load_map_and_units

    seen: set[tuple[int, int, int, int]] = set()
    for path in corpus_files:
        scenario = load_map_and_units(str(path))
        elevations, proj = elevations_and_proj(scenario)
        corner_rise = ig.corner_rise_px(elevations, proj, rule="average")
        h, w = elevations.shape
        for y in range(h):
            for x in range(w):
                d_nw, d_ne, d_sw, d_se = (
                    int(corner_rise[y, x]),
                    int(corner_rise[y, x + 1]),
                    int(corner_rise[y + 1, x]),
                    int(corner_rise[y + 1, x + 1]),
                )
                d_min = min(d_nw, d_ne, d_sw, d_se)
                seen.add((d_nw - d_min, d_ne - d_min, d_sw - d_min, d_se - d_min))
    # Print it, don't just bound it: the ceiling below is deliberately
    # generous, so a passing run says almost nothing on its own, and the
    # number is what any future maxsize decision has to be made against.
    # Run with -s to see it. Measured 2026-08-20: 318 keys on the quick
    # subset, 329 across the full corpus (--corpus-full) -- the subset
    # already covers almost the whole reachable key space, so the extra 17
    # files add 11 keys. Both are two orders of magnitude under the ceiling.
    print(f"\nsloped_quad_indices distinct keys across this corpus: {len(seen)}")
    # Generous ceiling, not a tight bound -- see the docstring above.
    assert len(seen) < 100_000, (
        f"{len(seen)} distinct sloped_quad_indices keys across the corpus -- "
        "revisit that function's maxsize=1024 against this number"
    )


@pytest.mark.corpus
@pytest.mark.parametrize("pct", [25, 50, 100, 200])
def test_sloped_output_stays_on_canvas_on_real_maps(corpus_files, pct):
    """Every tile's painted pixels land inside the canvas, and every column
    it paints is one contiguous run -- on REAL corpus terrain, at the
    shipped corner rule, across the elev_step_pct stops.

    What it actually pins, stated carefully because an earlier draft of this
    docstring claimed the wrong thing and the negative control caught it:

      1. **The caller convention, which is the load-bearing half.**
         sloped_quad_indices normalizes against min(corners) and folds the
         remainder into dst_y, so dst_y is explicitly NOT bounded below by 0
         (see its own docstring). Every caller therefore has to subtract that
         same d_min from base_y, and nothing in the type system says so.
         Verified non-vacuous 2026-08-23: dropping the `base_y -= d_min`
         below makes this fail at pct=200 on the first corpus file.
      2. **Canvas sizing against the rule.** canvas_h reserves exactly
         (max_elev - min_elev) * elev_step, and under "max" a corner really
         does reach the top of that -- a 480x480 corpus map's tallest rows
         land on `canvas_h - 1` exactly. Correct, but with zero slack, so a
         future rule or sizing change that overshoots by one pixel indexes
         off the canvas, where numpy fancy indexing wraps silently rather
         than raising. NOTE this is ordinary canvas sizing, NOT
         corner_headroom_px, which deliberately does not affect canvas_h at
         all (canvas_size_and_origin's own docstring) and which this test
         consequently does not exercise.
      3. **Per-column contiguity**, i.e. no hole inside a column.
         Deliberately NOT the inter-tile abutment property
         test_adjacent_tiles_abut_exactly pins: at pct=200 the max(raw_len,
         1) clamp lengthens runs and tiles genuinely DO overlap each other,
         which is documented and benign. A gap within one column would not
         be.

    Parametrized over pct rather than pinned to the default because the
    canvas is sized from elev_step, so the default is the one stop least
    likely to be where a sizing error shows up -- and pct=200 is where the
    negative control failed first.
    """
    from descape.render import SLOPE_CORNER_RULE, tile_pixels_for_map
    from descape.scenario_io import load_map_and_units

    # Contiguity and a tile's pixel EXTENT depend only on (tile_px,
    # normalized corners), not on where the tile sits -- so they are
    # resolved once per distinct key rather than once per tile. Real maps
    # produce a few hundred keys against hundreds of thousands of tiles
    # (see test_sloped_quad_indices_key_cardinality), which is what keeps
    # this affordable at corpus tier.
    def key_extent(tile_px, key, _cache={}):
        hit = _cache.get((tile_px, key))
        if hit is None:
            dst_y, dst_x, _sy, _sx, _uv = ig.sloped_quad_indices(tile_px, *key)
            counts = np.bincount(dst_x)
            lo = np.full(counts.size, np.iinfo(np.int64).max, dtype=np.int64)
            hi = np.full(counts.size, np.iinfo(np.int64).min, dtype=np.int64)
            np.minimum.at(lo, dst_x, dst_y)
            np.maximum.at(hi, dst_x, dst_y)
            used = counts > 0
            gapless = bool(((hi[used] - lo[used] + 1) == counts[used]).all())
            hit = (int(dst_y.min()), int(dst_y.max()), int(dst_x.min()), int(dst_x.max()), gapless)
            _cache[(tile_px, key)] = hit
        return hit

    checked = 0
    for path in corpus_files:
        scenario = load_map_and_units(str(path))
        mm = scenario.map_manager
        w, h = mm.map_width, mm.map_height
        tile_px = tile_pixels_for_map(w, h)
        elevations = np.zeros((h, w), dtype=np.int64)
        for tile in mm.terrain:
            elevations[tile.y, tile.x] = tile.elevation
        lo_e, hi_e = int(elevations.min()), int(elevations.max())
        proj = ig.canvas_size_and_origin(
            w, h, tile_px, lo_e, hi_e, elev_step_pct=pct, corner_headroom_steps=1
        )
        rise = ig.corner_rise_px(elevations, proj, rule=SLOPE_CORNER_RULE)
        for y in range(h):
            for x in range(w):
                d = (
                    int(rise[y, x]),
                    int(rise[y, x + 1]),
                    int(rise[y + 1, x]),
                    int(rise[y + 1, x + 1]),
                )
                d_min = min(d)
                dy0, dy1, dx0, dx1, gapless = key_extent(tile_px, tuple(v - d_min for v in d))
                base_x, base_y = ig.tile_screen_origin(x, y, 0, proj)
                base_y -= d_min  # the offset sloped_quad_indices' docstring requires
                assert gapless, f"column hole at {path.name} tile ({x},{y}) pct={pct}"
                assert 0 <= base_y + dy0 and base_y + dy1 < proj.canvas_h, (
                    f"{path.name} tile ({x},{y}) pct={pct}: rows "
                    f"{base_y + dy0}..{base_y + dy1} escape canvas_h={proj.canvas_h}"
                )
                assert 0 <= base_x + dx0 and base_x + dx1 < proj.canvas_w, (
                    f"{path.name} tile ({x},{y}) pct={pct}: cols "
                    f"{base_x + dx0}..{base_x + dx1} escape canvas_w={proj.canvas_w}"
                )
                checked += 1
    assert checked > 0, "no corpus tiles checked -- fixture resolved to nothing"
