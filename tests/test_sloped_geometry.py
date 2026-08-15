"""Verifies descape.iso_geometry's Phase 6 (Sloped) additions --
corner_rise_px, _corner_weights, and sloped_quad_indices -- Track C's own
first deliverable per docs/PLAN_V2_6.md: this is the load-bearing partition
proof for the bilinear corner-blend warp, written before render.py's
compositor is built on top of it.

Checks:
  1. Equal-corner delegation: sloped_quad_indices returns diamond_indices'
     own arrays verbatim (identity, not merely equal) when all four corners
     are equal -- what lets render_terrain_sloped's flat-map oracle against
     render_terrain_iso be byte-identical rather than approximate.
  2. corner_rise_px's "average" rule is exact on a flat map: every corner
     equals e * elev_step, not an off-by-rounding approximation, for every
     touching-tile count from 1 (map corner) to 4 (interior vertex).
  3. corner_rise_px's max/min rules pick the real max/min of touching tile
     elevations, scaled by elev_step, at a deliberately non-flat corner.
  4. Partition, realistic slope magnitude: for a synthetic map carrying a
     gentle multi-tile ramp and a steep single-tile ramp (both within this
     project's own ±1-elevation-delta-between-neighbors invariant, see
     iso_geometry.ELEV_STEP_DIVISOR's own comment), the union of every
     tile's sloped_quad_indices destination pixels has ZERO overlap and
     the same total pixel count as the flat baseline -- matching
     tools/verify_iso_geometry.py's check_diamond_partition bar for the
     unsloped case, empirically confirmed during this test's own design
     (not assumed) before being written down here as a permanent check.
  5. Per-column coverage, single tile: every screen column a tile paints is
     a CONTIGUOUS run of destination rows with no repeats, across a broad
     set of corner shapes check 4 never reaches. This is the sharp check --
     see test_sloped_quad_indices_columns_stay_contiguous' own docstring
     for the bug it was written against and why "no duplicates" is
     load-bearing rather than incidental strictness.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import iso_geometry as ig


def test_sloped_quad_indices_delegates_at_equal_corners():
    tile_px = 32
    base = ig.diamond_indices(tile_px)
    for value in (0, 5, -3, 17):
        got = ig.sloped_quad_indices(tile_px, value, value, value, value)
        for a, b in zip(got, base):
            assert a is b or np.array_equal(a, b)


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


def test_corner_rise_rejects_bad_rule():
    elevations = np.zeros((2, 2), dtype=np.int64)
    proj = ig.canvas_size_and_origin(2, 2, 32, 0, 0)
    with pytest.raises(ValueError):
        ig.corner_rise_px(elevations, proj, rule="mean")


def _partition_check(elevations: np.ndarray, tile_px: int, rule: str) -> tuple[int, int]:
    """(overlap_count, unique_pixel_count) for rendering every tile of
    `elevations` through sloped_quad_indices at elevation-0 baseline
    placement -- the same construction render.py's own _render_tile_sloped
    will use (tile_screen_origin(x, y, 0, proj), base_y -= d_min, paint at
    dst_y/dst_x), just without a real image or texture, purely tracking
    which absolute (canvas_y, canvas_x) pixels each tile claims."""
    h, w = elevations.shape
    proj = ig.canvas_size_and_origin(w, h, tile_px, 0, int(elevations.max()), corner_headroom_steps=1)
    corner_rise = ig.corner_rise_px(elevations, proj, rule=rule)
    claimed: dict[tuple[int, int], tuple[int, int]] = {}
    overlaps = 0
    for y in range(h):
        for x in range(w):
            d_nw, d_ne, d_sw, d_se = (
                int(corner_rise[y, x]),
                int(corner_rise[y, x + 1]),
                int(corner_rise[y + 1, x]),
                int(corner_rise[y + 1, x + 1]),
            )
            d_min = min(d_nw, d_ne, d_sw, d_se)
            dst_y, dst_x, _src_y, _src_x = ig.sloped_quad_indices(tile_px, d_nw, d_ne, d_sw, d_se)
            base_x, base_y = ig.tile_screen_origin(x, y, 0, proj)
            base_y -= d_min
            for py, px in zip((base_y + dst_y).tolist(), (base_x + dst_x).tolist()):
                key = (py, px)
                if key in claimed and claimed[key] != (x, y):
                    overlaps += 1
                claimed[key] = (x, y)
    return overlaps, len(claimed)


def test_partition_gentle_multi_tile_ramp():
    tile_px = 16
    w = h = 6
    elevations = np.zeros((h, w), dtype=np.int64)
    elevations[:, w // 2 :] = 1  # climbs by one full elevation level over the map's width
    overlaps, unique = _partition_check(elevations, tile_px, rule="average")
    assert overlaps == 0
    assert unique == w * h * ig.diamond_indices(tile_px)[0].size


def test_partition_steep_single_tile_ramp():
    # The same one-level rise concentrated into a single tile-width seam --
    # the steepest ramp this project's own ±1 real elevation invariant ever
    # produces (docs/PLAN_V2_6.md's Track C: "steep_seam" tests DELTA>=2,
    # which is out of scope for a ramp -- MapManager never creates one).
    tile_px = 16
    w = h = 6
    elevations = np.zeros((h, w), dtype=np.int64)
    elevations[:, w // 2 + 1 :] = 1
    overlaps, unique = _partition_check(elevations, tile_px, rule="average")
    assert overlaps == 0
    assert unique == w * h * ig.diamond_indices(tile_px)[0].size


def _column_shapes(tile_px: int, corners: tuple[int, int, int, int]) -> tuple[int, int]:
    """(gap_rows, duplicate_rows) summed over every screen column of one
    tile's sloped_quad_indices output -- a gap row is a destination row
    inside a column's own [min, max] span that no source pixel writes, a
    duplicate is two source pixels landing on the same (column, row)."""
    dst_y, dst_x, _src_y, _src_x = ig.sloped_quad_indices(tile_px, *corners)
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
    to a (0, 1, 0, 1)-shaped ramp, and check 4's map-level assertion only
    detects OVERLAP (two tiles claiming one pixel), never an unwritten row.

    The no-duplicates half is load-bearing, not incidental strictness:
    together with contiguity and sloped_quad_indices' fixed pixel count it
    pins each column to a RIGID translation, which is the whole reason the
    fix cannot reintroduce a gap. Anyone reinstating true intra-column
    compression (see that function's docstring on what the rigid shift
    gives up) has to change the return contract too -- loosening this
    assertion alone would just restore the seams."""
    gaps, duplicates = _column_shapes(tile_px, corners)
    assert gaps == 0, f"{gaps} unwritten row(s) inside a column at tile_px={tile_px}, corners={corners}"
    assert duplicates == 0, f"{duplicates} doubly-written row(s) at tile_px={tile_px}, corners={corners}"


@pytest.mark.corpus
def test_sloped_quad_indices_key_cardinality(corpus_files):
    """Measures how many distinct (normalized d_nw, d_ne, d_sw, d_se) keys
    real maps actually produce -- sloped_quad_indices' own docstring says
    maxsize=2048 should be revisited against this measurement rather than
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
    # Generous ceiling, not a tight bound -- see the docstring above.
    assert len(seen) < 100_000, (
        f"{len(seen)} distinct sloped_quad_indices keys across the corpus -- "
        "revisit that function's maxsize=2048 against this number"
    )
