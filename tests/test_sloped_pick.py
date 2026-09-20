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
  5. The pick-plane memo is bounded, and no source-state change can leave a
     stale plane behind: invalidate_region()/set_unit_filter() drop the lot,
     and patch() either drops a plane (over PICK_PLANE_PATCH_MAX_FRACTION)
     or rewrites the intersected sub-rect in place.
  6. A plane patch() rewrote in place is byte-identical to a fresh
     composite_ids_rect_sloped over that plane's own clipped rect. This is
     B8's whole correctness bar: an under-covered rewrite leaves the plane
     reporting the pre-edit tile, which is a wrong click target with no
     crash, so the oracle compares the WHOLE plane rather than the
     rewritten sub-rect.

Fixture note, and it is deliberate: a LOW-AMPLITUDE BUMPY field is the
visually worst case, not the mildest. It puts a transition at nearly
every tile boundary, so a
geometry defect covers the whole area. A tall hill has more total error but
confines it to fewer edges. Judge on the bumpy field.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import iso_geometry, render
from descape.elevation_tools import set_tile_elevation
from descape.render import (
    PICK_ID_NONE,
    composite_ids_rect_sloped,
    dirty_screen_bbox_sloped,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import (
    MAX_PICK_PLANES,
    PICK_PLANE_PATCH_MAX_FRACTION,
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

    for raw_sx, raw_sy in zip(*_sample_pixels(proj), strict=True):
        sx, sy = int(raw_sx), int(raw_sy)
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
    for raw_sx, raw_sy in zip(*_sample_pixels(proj, n=1500), strict=True):
        sx, sy = int(raw_sx), int(raw_sy)
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
    resolving against geometry that is no longer on screen.

    A WHOLE-CANVAS bbox, which is why both arms still assert a wholesale
    drop after B8: invalidate_region() drops unconditionally, and for
    patch() every resident plane is 100% covered, far above
    PICK_PLANE_PATCH_MAX_FRACTION, so every one is dropped rather than
    rewritten. The partial-coverage cases patch() now handles instead are
    check 6's own tests below."""
    scenario = _bumpy_scenario()
    _elev, corner_rise, proj, tile_px = _sloped_state(scenario)
    cache = SlopedChunkCache(scenario, _elev, corner_rise, proj, tile_px, chunk_px=128)

    cache.pick_tile(proj.canvas_w // 2, proj.canvas_h // 2)
    assert cache._pick_planes
    getattr(cache, method)((0, 0, proj.canvas_w, proj.canvas_h))
    assert not cache._pick_planes


def _edit_and_bbox(scenario, elevations, proj, ex: int, ey: int, delta: int):
    """One real elevation edit, returning the (bbox, elevation_changed) pair
    ViewerWindow._apply_dirty hands patch().

    dirty_screen_bbox_sloped mutates `elevations` in place by contract, and
    _sloped_state() hands back the same array object the cache was built
    with, which is what lets the cache's own corner_rise rebuild inside
    patch() see this edit at all."""
    mm = scenario.map_manager
    before = [int(t.elevation) for t in mm.terrain]
    set_tile_elevation(mm, ex, ey, int(mm.get_tile(ex, ey).elevation) + delta)
    dirty = [i for i, t in enumerate(mm.terrain) if int(t.elevation) != before[i]]
    assert dirty, "the fixture edit changed no tile, so the test below would prove nothing"
    elevation_changed: set = set()
    bbox = dirty_screen_bbox_sloped(
        scenario, dirty, elevations, proj, with_units=True, elevation_changed=elevation_changed
    )
    assert bbox is not None, "a legal in-range edit must not decline the incremental path"
    assert elevation_changed, "an elevation edit must report a non-empty elevation_changed"
    return bbox, elevation_changed


def _plane_rect(cache, key) -> tuple[int, int, int, int]:
    """A resident plane's own pixel rect, taken from .shape rather than
    chunk_px so an edge plane clipped by canvas_dims() reads correctly."""
    plane = cache._pick_planes[key]
    px0, py0 = key[0] * cache.chunk_px, key[1] * cache.chunk_px
    return px0, py0, px0 + plane.shape[1], py0 + plane.shape[0]


def _plane_oracle(cache, key) -> np.ndarray:
    """Check 6's bar: what that plane's ids must be, composited fresh
    against the cache's post-patch corner_rise."""
    px0, py0, px1, py1 = _plane_rect(cache, key)
    return composite_ids_rect_sloped(
        cache.scenario, px0, py0, px1, py1, cache.corner_rise, cache.proj, cache.tile_px
    )


def _coverage(cache, key, bbox) -> float:
    """The share of that plane the bbox covers, i.e. what patch() compares
    against PICK_PLANE_PATCH_MAX_FRACTION."""
    px0, py0, px1, py1 = _plane_rect(cache, key)
    bx0, by0, bx1, by1 = bbox
    w = max(0, min(bx1, px1) - max(bx0, px0))
    h = max(0, min(by1, py1) - max(by0, py0))
    return w * h / ((px1 - px0) * (py1 - py0))


def test_patch_rewrites_a_resident_plane_in_place():
    """Check 6a: the whole point of B8. The plane object must survive the
    patch (a drop-and-rebuild would satisfy the oracle while paying exactly
    the 13.5ms this step exists to remove), its contents must match a fresh
    composite, and they must differ from the pre-edit copy, without which
    the oracle could be comparing two identical no-ops."""
    scenario = _bumpy_scenario()
    elevations, corner_rise, proj, tile_px = _sloped_state(scenario)
    cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, chunk_px=1024)

    sx, sy = proj.canvas_w // 2, proj.canvas_h // 2
    tile = cache.pick_tile(sx, sy)
    assert tile is not None, "the canvas centre must be on-map for this fixture"
    key = (sx // cache.chunk_px, sy // cache.chunk_px)
    plane = cache._pick_planes[key]
    before = plane.copy()

    bbox, elevation_changed = _edit_and_bbox(scenario, elevations, proj, tile[0], tile[1], 1)
    covered = _coverage(cache, key, bbox)
    assert 0 < covered <= PICK_PLANE_PATCH_MAX_FRACTION, (
        f"the fixture edit covers {covered:.0%} of the plane, so patch() would DROP it. "
        "This test would then be exercising the drop path, not the rewrite it is about"
    )

    cache.patch(bbox, elevation_changed=elevation_changed)

    assert cache._pick_planes.get(key) is plane, "the plane was rebuilt, not patched in place"
    assert not np.array_equal(plane, before), "the edit moved no ids, so the oracle below proves nothing"
    assert np.array_equal(plane, _plane_oracle(cache, key))


def test_patch_rewrites_both_planes_a_bbox_straddles():
    """Check 6b: the partial-coverage path on TWO planes at once, which
    is what a drag along a chunk seam does and where the per-plane slice
    arithmetic can silently go wrong (an offset taken from the bbox rather
    than from each plane's own origin still lands inside the array, so it
    corrupts rather than raises)."""
    scenario = _bumpy_scenario()
    elevations, corner_rise, proj, tile_px = _sloped_state(scenario)
    chunk_px = 1024
    cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, chunk_px=chunk_px)

    seam_x = (proj.canvas_w // 2 // chunk_px) * chunk_px
    sy = proj.canvas_h // 2
    tile = cache.pick_tile(seam_x, sy)
    assert tile is not None
    cache.pick_tile(seam_x - 1, sy)  # warm the plane on the other side too
    left, right = ((seam_x - 1) // chunk_px, sy // chunk_px), (seam_x // chunk_px, sy // chunk_px)
    assert left != right and left in cache._pick_planes and right in cache._pick_planes
    planes = {key: cache._pick_planes[key] for key in (left, right)}
    before = {key: plane.copy() for key, plane in planes.items()}

    bbox, elevation_changed = _edit_and_bbox(scenario, elevations, proj, tile[0], tile[1], 1)
    for key in (left, right):
        covered = _coverage(cache, key, bbox)
        assert 0 < covered <= PICK_PLANE_PATCH_MAX_FRACTION, f"{key} is covered {covered:.0%}, not straddled"

    cache.patch(bbox, elevation_changed=elevation_changed)

    for key in (left, right):
        assert cache._pick_planes.get(key) is planes[key], f"{key} was rebuilt, not patched in place"
        assert np.array_equal(planes[key], _plane_oracle(cache, key)), f"{key} disagrees with a fresh composite"
    assert any(not np.array_equal(planes[key], before[key]) for key in (left, right)), "the edit moved no ids"


def test_patch_rewrites_an_edge_plane_clipped_by_the_canvas():
    """Check 6c: an edge plane is RAGGED (_pick_plane clips it to
    canvas_dims), so its high edge must come from .shape, never from
    chunk_px. The bbox here deliberately overshoots the canvas, which is
    free (over-covering only recomposites more): with the chunk_px slip the
    sub-rect composites past the canvas and the slice-assign raises on the
    shape mismatch instead of silently mis-writing."""
    scenario = _bumpy_scenario()
    elevations, corner_rise, proj, tile_px = _sloped_state(scenario)
    chunk_px = 1024
    cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, chunk_px=chunk_px)

    cy = (proj.canvas_h - 1) // chunk_px
    sy = (cy * chunk_px + proj.canvas_h) // 2
    sx = proj.canvas_w // 2
    tile = cache.pick_tile(sx, sy)
    assert tile is not None, "the bottom chunk row's centre must be on-map for this fixture"
    key = (sx // chunk_px, cy)
    plane = cache._pick_planes[key]
    assert plane.shape[0] != chunk_px, "this fixture's bottom chunk row is not clipped, so the check is toothless"

    bbox, elevation_changed = _edit_and_bbox(scenario, elevations, proj, tile[0], tile[1], 1)
    bbox = (bbox[0], bbox[1], bbox[2], proj.canvas_h + tile_px)
    covered = _coverage(cache, key, bbox)
    assert 0 < covered <= PICK_PLANE_PATCH_MAX_FRACTION, f"the edge plane is covered {covered:.0%}, so it is dropped"

    cache.patch(bbox, elevation_changed=elevation_changed)

    assert cache._pick_planes.get(key) is plane, "the edge plane was rebuilt, not patched in place"
    assert np.array_equal(plane, _plane_oracle(cache, key))


def test_an_empty_elevation_changed_leaves_a_warm_plane_untouched(monkeypatch):
    """Check 6d: the terrain-paint-only case, which the findings call most
    edits. A plane's ids read corner_rise, proj, tile_px and the map dims
    alone, so an edit that moved no elevation cannot move an id and must
    cost nothing at all. None still means "unknown" and must not take this
    shortcut: over a whole-canvas bbox it drops, exactly as check 5b pins."""
    scenario = _bumpy_scenario()
    elevations, corner_rise, proj, tile_px = _sloped_state(scenario)
    cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, chunk_px=1024)

    sx, sy = proj.canvas_w // 2, proj.canvas_h // 2
    assert cache.pick_tile(sx, sy) is not None
    key = (sx // cache.chunk_px, sy // cache.chunk_px)
    plane = cache._pick_planes[key]
    before = plane.copy()

    # Counted, not just compared: a rewrite against an unchanged corner_rise
    # produces the same bytes, so equality alone would pass the wasted work.
    calls = []
    real = render.composite_ids_rect_sloped
    monkeypatch.setattr(
        render, "composite_ids_rect_sloped", lambda *a, **k: (calls.append(a[1:5]), real(*a, **k))[1]
    )
    cache.patch((sx - 64, sy - 64, sx + 64, sy + 64), elevation_changed=set())
    assert not calls, f"an empty elevation_changed recomposited {len(calls)} sub-rect(s) it cannot have invalidated"
    assert cache._pick_planes.get(key) is plane, "an empty elevation_changed must not touch the memo"
    assert np.array_equal(plane, before), "an empty elevation_changed rewrote a plane it cannot have invalidated"

    cache.patch((0, 0, proj.canvas_w, proj.canvas_h), elevation_changed=None)
    assert not cache._pick_planes, "an unknown elevation_changed over the whole canvas must drop every plane"
