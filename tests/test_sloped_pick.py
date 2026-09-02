"""Verifies Sloped mode's hit-testing backend -- Track C4's
render._render_tile_sloped_ids/composite_ids_rect_sloped and
SlopedChunkCache.pick_tile/_pick_plane.

The bar this suite has to hold, and it is not "the pick looks sensible": the
id plane must agree with the pixels the COLOUR compositor actually painted,
pixel for pixel. Every quantity deciding which pixels a tile claims is copied
from _render_tile_sloped into _render_tile_sloped_ids rather than re-derived,
so a mistake there is a SILENT disagreement between what the user sees and
what their click edits -- never a crash. That is what checks 2 and 3 below
exist to catch, and why check 2 recomputes the geometry from iso_geometry
directly instead of calling the compositor it is supposed to be auditing.

Checks:
  1. A chunk-local pick_tile() agrees with a whole-canvas id composite over
     random pixels -- the chunk-boundary/offset failure, mirroring the
     chunk-assembly-vs-full-render independence tests/test_sloped_chunks.py
     already relies on.
  2. A whole-canvas id composite agrees with an INDEPENDENT oracle built
     straight from iso_geometry (check _pick_oracle). This is the one that
     pins the -d_min normalization term.
  3. Exhaustive partition: no canvas pixel is claimed by two tiles, and the
     canvas interior has no unclaimed holes.
  4. Uniform elevation delegates to iso_geometry.screen_to_tile exactly.
     Honest about its reach: sloped_quad_indices returns diamond_indices'
     arrays verbatim there, so this exercises ZERO resample code and proves
     nothing about the sloped branch. Kept because it pins the delegation,
     not because it covers geometry.
  5. The pick-plane memo is bounded, and is dropped by every source-state
     change (patch/invalidate_region/set_unit_filter).

Fixture note, and it is deliberate: a LOW-AMPLITUDE BUMPY field is the
visually worst case, not the mildest. It puts a transition at nearly
every tile boundary, so a
geometry defect covers the whole area. A tall hill has more total error but
confines it to fewer edges. Judge on the bumpy field.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import iso_geometry
from descape.render import (
    PICK_ID_NONE,
    composite_ids_rect_sloped,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import (
    MAX_PICK_PLANES,
    SlopedChunkCache,
)
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

RNG_SEED = 20260820
SAMPLE_PIXELS = 4000


def _scenario(elevation_of) -> object:
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = elevation_of(tile.x, tile.y)
    return scenario


def _bumpy_scenario():
    """A 1-level bumpy field -- see this module's docstring for why low
    amplitude is the demanding case rather than the gentle one. Blocked
    2x3 rather than a per-tile checkerboard so runs of equal corners and
    runs of differing ones both occur."""
    return _scenario(lambda x, y: (x // 2 + y // 3) % 2)


def _flat_scenario():
    return _scenario(lambda x, y: 0)


def _sloped_state(scenario):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    return elevations, corner_rise, proj, tile_px


def _pick_oracle(scenario, corner_rise, proj, tile_px):
    """(ids, counts) over the whole reference canvas, recomputed from
    iso_geometry alone -- NOT through render.py's compositor, which is the
    code under test.

    This restates the painting contract the compositor is supposed to
    implement: paint in depth_order, place at tile_screen_origin(x, y, 0)
    (elevation 0, because corner_rise already encodes the absolute
    elevation-to-pixel scale and adding a second term would double-count
    it), shift base_y by -d_min to match sloped_quad_indices' own
    normalization, and drop out-of-canvas pixels. `counts` additionally
    tallies how many tiles claim each pixel, which is what makes the
    partition check exhaustive rather than a spot check."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    ids = np.full((proj.canvas_h, proj.canvas_w), PICK_ID_NONE, dtype=np.int32)
    counts = np.zeros((proj.canvas_h, proj.canvas_w), dtype=np.int32)

    for x, y in iso_geometry.depth_order(w, h):
        d_nw = int(corner_rise[y, x])
        d_ne = int(corner_rise[y, x + 1])
        d_sw = int(corner_rise[y + 1, x])
        d_se = int(corner_rise[y + 1, x + 1])
        d_min = min(d_nw, d_ne, d_sw, d_se)

        base_x, base_y = iso_geometry.tile_screen_origin(x, y, 0, proj)
        base_y -= d_min

        dst_y, dst_x, _sy, _sx, _uv = iso_geometry.sloped_quad_indices(tile_px, d_nw, d_ne, d_sw, d_se)
        ay, ax = base_y + dst_y, base_x + dst_x
        ok = (ay >= 0) & (ay < ids.shape[0]) & (ax >= 0) & (ax < ids.shape[1])
        ay, ax = ay[ok], ax[ok]
        ids[ay, ax] = y * w + x
        counts[ay, ax] += 1
    return ids, counts


def _full_canvas_ids(scenario, corner_rise, proj, tile_px) -> np.ndarray:
    return composite_ids_rect_sloped(scenario, 0, 0, proj.canvas_w, proj.canvas_h, corner_rise, proj, tile_px)


def _sample_pixels(proj, n: int = SAMPLE_PIXELS):
    rng = np.random.default_rng(RNG_SEED)
    xs = rng.integers(0, proj.canvas_w, size=n)
    ys = rng.integers(0, proj.canvas_h, size=n)
    return xs, ys


def test_chunk_local_pick_agrees_with_the_whole_canvas_plane():
    """Check 1 -- the chunk-boundary/offset failure. pick_tile() composites
    one clipped chunk at a nonzero (x0, y0) offset; the oracle composites
    the whole canvas at offset (0, 0). An offset handled wrongly in
    _render_tile_sloped_ids shows up here and essentially nowhere else."""
    scenario = _bumpy_scenario()
    _elev, corner_rise, proj, tile_px = _sloped_state(scenario)
    cache = SlopedChunkCache(scenario, _elev, corner_rise, proj, tile_px, chunk_px=128)
    full = _full_canvas_ids(scenario, corner_rise, proj, tile_px)
    map_w = scenario.map_manager.map_width

    for sx, sy in zip(*_sample_pixels(proj)):
        sx, sy = int(sx), int(sy)
        expected_id = int(full[sy, sx])
        expected = None if expected_id == PICK_ID_NONE else (expected_id % map_w, expected_id // map_w)
        assert cache.pick_tile(sx, sy) == expected, f"disagreement at ({sx}, {sy})"


def test_pick_plane_agrees_with_an_independent_geometry_oracle():
    """Check 2 -- the load-bearing one. Audits the compositor against
    iso_geometry directly, so a wrong base_y/base_x term (notably dropping
    -d_min) fails here even though every path through render.py would still
    agree with itself."""
    scenario = _bumpy_scenario()
    _elev, corner_rise, proj, tile_px = _sloped_state(scenario)
    oracle_ids, _counts = _pick_oracle(scenario, corner_rise, proj, tile_px)
    assert np.array_equal(_full_canvas_ids(scenario, corner_rise, proj, tile_px), oracle_ids)


def test_no_canvas_pixel_is_claimed_by_two_tiles():
    """Check 3a -- overlaps. Overlaps hide about half the error and are
    invisible on screen (they just overpaint), so `overlaps == 0` is half
    the correctness bar, not decoration. The seam resample made sloped
    coverage an exact partition; this is the pick plane inheriting that."""
    scenario = _bumpy_scenario()
    _elev, corner_rise, proj, tile_px = _sloped_state(scenario)
    _ids, counts = _pick_oracle(scenario, corner_rise, proj, tile_px)
    assert counts.max() <= 1, f"{int((counts > 1).sum())} canvas pixels claimed by more than one tile"


def test_canvas_interior_has_no_unclaimed_holes():
    """Check 3b -- gaps. A rect at the canvas center is comfortably inside
    the map's diamond silhouette on a full-size template, so every pixel in
    it must belong to some tile. This is the check that would have caught
    the pre-resample 1px seam lattice."""
    scenario = _bumpy_scenario()
    _elev, corner_rise, proj, tile_px = _sloped_state(scenario)
    ids = _full_canvas_ids(scenario, corner_rise, proj, tile_px)

    half_w, half_h = proj.canvas_w // 16, proj.canvas_h // 16
    cx, cy = proj.canvas_w // 2, proj.canvas_h // 2
    interior = ids[cy - half_h : cy + half_h, cx - half_w : cx + half_w]
    assert interior.size > 0
    assert not (interior == PICK_ID_NONE).any(), f"{int((interior == PICK_ID_NONE).sum())} unclaimed interior px"


def test_uniform_elevation_delegates_to_screen_to_tile():
    """Check 4 -- the delegation guard. On a flat map sloped_quad_indices
    hands back diamond_indices' arrays verbatim, so the pick must match
    Stepped's analytic inverse exactly. Reach is genuinely narrow: this
    exercises no resample code at all (see this module's docstring)."""
    scenario = _flat_scenario()
    elevations, corner_rise, proj, tile_px = _sloped_state(scenario)
    cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, chunk_px=128)

    checked = 0
    for sx, sy in zip(*_sample_pixels(proj, n=1500)):
        sx, sy = int(sx), int(sy)
        expected = iso_geometry.screen_to_tile(sx, sy, elevations, proj)
        if expected is None:
            continue
        assert cache.pick_tile(sx, sy) == expected, f"disagreement at ({sx}, {sy})"
        checked += 1
    assert checked > 100, f"only {checked} on-map samples -- the assertion above never really ran"


def test_off_canvas_positions_pick_nothing():
    scenario = _bumpy_scenario()
    _elev, corner_rise, proj, tile_px = _sloped_state(scenario)
    cache = SlopedChunkCache(scenario, _elev, corner_rise, proj, tile_px)
    for sx, sy in ((-1, 10), (10, -1), (proj.canvas_w, 10), (10, proj.canvas_h)):
        assert cache.pick_tile(sx, sy) is None


@pytest.mark.parametrize(
    "corners",
    [(0, 0, 0, 0), (0, 8, 0, 8), (8, 0, 8, 0), (0, 4, 4, 8), (6, 0, 0, 6)],
    ids=["flat", "east-rising", "west-rising", "twisted", "saddle"],
)
def test_tile_outline_spans_exactly_the_painted_pixels(corners):
    """The hover outline must bound precisely the pixels the tile claims --
    an outline that disagrees with the pick plane is a highlight promising
    an edit somewhere the click would not land.

    Independent by construction: the outline comes from _sloped_column_runs'
    run_start/run_len, while the expected span is measured off
    sloped_quad_indices' fully expanded dst_y/dst_x. Those are different
    arrays derived at different stages, so a mistake in the expansion or in
    the outline shows up as a disagreement rather than cancelling out."""
    tile_px = 32
    d_nw, d_ne, d_sw, d_se = corners
    dst_y, dst_x, _sy, _sx, _uv = iso_geometry.sloped_quad_indices(tile_px, d_nw, d_ne, d_sw, d_se)
    points = iso_geometry.sloped_tile_outline(tile_px, d_nw, d_ne, d_sw, d_se)

    # The ring is 4 points per column: the top edge as (c, y0), (c+1, y0)
    # per column ascending, then the bottom edge reversed. Grouping by x
    # alone would NOT work -- x = c+1 is shared between a column's own right
    # edge and its neighbour's left edge -- so index the staircase instead.
    n = len(points) // 4
    top_pts, bottom_pts = points[: 2 * n], points[2 * n :][::-1]
    spans: dict[int, tuple[int, int]] = {}
    for i in range(n):
        col, y0 = top_pts[2 * i]
        col_again, y1 = bottom_pts[2 * i]
        assert col == col_again, "top and bottom staircases disagree about column order"
        spans[col] = (y0, y1)

    for col in np.unique(dst_x):
        rows = dst_y[dst_x == col]
        lo, hi = int(rows.min()), int(rows.max())
        span = spans.get(int(col))
        assert span is not None, f"column {col} is painted but absent from the outline"
        # The bottom edge is exclusive (y1 = run_start + run_len), one past
        # the last painted row -- which is what makes the polygon enclose
        # that row rather than bisect it.
        assert span == (lo, hi + 1), f"column {col}: outline {span} != painted ({lo}, {hi + 1})"


def test_pick_plane_memo_is_bounded():
    """Check 5a -- the memo is a small side LRU, deliberately outside the
    colour cache's own LRU and byte budget (C plan decision 7)."""
    scenario = _bumpy_scenario()
    _elev, corner_rise, proj, tile_px = _sloped_state(scenario)
    cache = SlopedChunkCache(scenario, _elev, corner_rise, proj, tile_px, chunk_px=64)

    for i in range(MAX_PICK_PLANES + 3):
        cache.pick_tile(min(i * 64, proj.canvas_w - 1), 0)
    assert len(cache._pick_planes) <= MAX_PICK_PLANES
    assert cache._cache_bytes == 0, "pick planes must not be billed to the colour cache's byte budget"


@pytest.mark.parametrize("method", ["patch", "invalidate_region"])
def test_source_state_changes_drop_the_memo(method):
    """Check 5b -- a memo outliving an edit is exactly how a click starts
    resolving against geometry that is no longer on screen."""
    scenario = _bumpy_scenario()
    _elev, corner_rise, proj, tile_px = _sloped_state(scenario)
    cache = SlopedChunkCache(scenario, _elev, corner_rise, proj, tile_px, chunk_px=128)

    cache.pick_tile(proj.canvas_w // 2, proj.canvas_h // 2)
    assert cache._pick_planes
    getattr(cache, method)((0, 0, proj.canvas_w, proj.canvas_h))
    assert not cache._pick_planes
