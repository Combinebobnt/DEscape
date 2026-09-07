"""Phase B-D-a of the mip-level plan -- pure bookkeeping tests for the
mip-level plumbing added to descape.render_cache._ChunkCacheBase,
IsoChunkCache and FlatChunkCache. Phase a's
own level set is still {0: reference} (no other levels are enumerated
yet -- that's Phase B-D-b), so this file deliberately owns no full-canvas
byte-identity assertion: that bar stays with tools/verify_iso_chunks.py /
tests/test_flat_chunks.py, unedited by this phase, and is re-exercised at
real non-zero mips by tests/test_mip_geometry.py once Phase B-D-b lands.

Every test here is bookkeeping arithmetic plus a handful of chunk touches
against the small (120x120, unit-free) fixture template -- milliseconds,
not a corpus pass.

What Phase B-D-a genuinely tests, vs. what it structurally cannot: the
level set is [0]; mip_scale(0) == 1.0 exactly;
render_rect()'s trailing mip=0 default matches an explicit call; the byte
budget's admitted set is arithmetically equal to the old chunk-count
formula; explicit max_chunks keeps capping strictly by count with no byte
bound, mirroring the exact call shapes tools/verify_iso_chunks.py and
tests/test_flat_chunks.py already depend on; _cache_bytes bookkeeping
stays consistent across insert/evict/invalidate_region/patch (the
invalidate_region decrement is the easiest line of this whole phase to
forget silently); and _bbox_to_level is the identity at the reference
level (it cannot be tested for real until Phase B-D-b adds a second
level).
"""

from __future__ import annotations

import pytest

from descape.render import elevations_and_proj, tile_pixels_for_map
from descape.render_cache import DEFAULT_CHUNK_PX, FlatChunkCache, IsoChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.scenario_io import load_map_and_units


def _build_iso_cache(max_chunks=None, chunk_px=DEFAULT_CHUNK_PX) -> IsoChunkCache:
    scenario = load_map_and_units(FIXTURE_PATH)
    tile_px = tile_pixels_for_map(scenario.map_manager.map_width, scenario.map_manager.map_height)
    elevations, proj = elevations_and_proj(scenario)
    return IsoChunkCache(scenario, elevations, proj, tile_px, chunk_px=chunk_px, max_chunks=max_chunks)


def _build_flat_cache(max_chunks=None, chunk_px=DEFAULT_CHUNK_PX) -> FlatChunkCache:
    scenario = load_map_and_units(FIXTURE_PATH)
    tile_px = tile_pixels_for_map(scenario.map_manager.map_width, scenario.map_manager.map_height)
    return FlatChunkCache(scenario, tile_px, chunk_px=chunk_px, max_chunks=max_chunks)


# Both cache types share every bit of the bookkeeping under test (it all
# lives in _ChunkCacheBase) -- parametrizing over the two constructors lets
# every test below exercise both without duplicating the test bodies.
_BUILDERS = [_build_iso_cache, _build_flat_cache]
_BUILDER_IDS = ["iso", "flat"]


@pytest.fixture(params=_BUILDERS, ids=_BUILDER_IDS)
def cache(request):
    return request.param()


def test_level_zero_is_always_the_reference(cache) -> None:
    """True regardless of how many other levels the real per-level
    projection set (Phase B-D-b) enumerates for this particular fixture/
    elev_step_pct combination -- 0 must always be present and must always
    mean the reference tile_px (D2)."""
    assert 0 in cache.mip_levels()
    assert cache.mip_tile_px(0) == cache.tile_px
    assert cache.mip_scale(0) == 1.0  # exact float comparison -- both operands are the same int


def test_mip_for_scale_never_returns_a_level_outside_the_enumerated_set(cache) -> None:
    levels = cache.mip_levels()
    for scale in (0.001, 0.1, 1.0, 3.7, 64.0, 0.0, -5.0):
        assert cache.mip_for_scale(scale) in levels
    # Non-positive scale is a defensive floor (not a real caller's input):
    # it must land on the coarsest available level, never raise.
    assert cache.mip_for_scale(0.0) == levels[0]
    assert cache.mip_for_scale(-5.0) == levels[0]


def test_mip_levels_plumbing_with_a_synthetic_single_level_table(cache) -> None:
    """Phase B-D-a's own bookkeeping (mip_levels/mip_scale/mip_for_scale)
    exercised directly against a single-entry table, decoupled from
    whichever real per-level set this fixture's elev_step_pct happens to
    produce (Phase B-D-b) -- this is the code path a genuinely
    single-level configuration (e.g. reference tile_px=16 at
    elev_step_pct=10, see tests/test_mip_geometry.py) exercises for real,
    pinned here without needing to construct that exact scenario."""
    cache._init_mip_levels({0: cache.tile_px})
    assert cache.mip_levels() == [0]
    assert cache.mip_scale(0) == 1.0
    for scale in (0.001, 0.1, 1.0, 3.7, 64.0, 0.0, -5.0):
        assert cache.mip_for_scale(scale) == 0


def test_default_mip_matches_explicit_zero(cache) -> None:
    canvas_w, canvas_h = cache.canvas_dims()
    # A deliberately non-chunk-aligned rect, not just (0,0,canvas): the
    # trailing default has to thread through render_rect's whole clipping
    # path, not just its first argument.
    x0, y0 = min(17, canvas_w - 1), min(23, canvas_h - 1)
    x1, y1 = min(canvas_w, x0 + 200), min(canvas_h, y0 + 150)
    default = cache.render_rect(x0, y0, x1, y1)
    explicit = cache.render_rect(x0, y0, x1, y1, mip=0)
    assert default.shape == explicit.shape
    assert (default == explicit).all()


def test_canvas_dims_default_matches_explicit_zero(cache) -> None:
    assert cache.canvas_dims() == cache.canvas_dims(mip=0)


def test_bbox_to_level_is_identity_at_the_reference(cache) -> None:
    canvas_w, canvas_h = cache.canvas_dims()
    bboxes = [
        (0, 0, canvas_w, canvas_h),
        (0, 0, 0, 0),  # zero-area
        (canvas_w - 1, canvas_h - 1, canvas_w, canvas_h),  # canvas high edge
        (17, 23, 200, 150),
    ]
    for bbox in bboxes:
        assert cache._bbox_to_level(0, bbox) == bbox


def test_byte_budget_admits_exactly_the_old_count_budget(cache) -> None:
    """No pixels needed: at the reference level chunks tile the canvas
    exactly (each clipped to canvas bounds at the high edge), so the sum
    of every chunk's own byte size equals canvas_w * canvas_h * 3 -- the
    same total a fully-warmed cache under the OLD count formula
    (ceil(W/cp) * ceil(H/cp) chunks) would hold, since that formula's
    chunk count times each chunk's own clipped size sums to the same
    canvas area. This proves the two budgets admit the same set for a
    single-level cache without compositing a single pixel."""
    canvas_w, canvas_h = cache.canvas_dims()
    assert cache.max_chunks is None
    assert cache.max_bytes == canvas_w * canvas_h * 3

    old_n_cx = (canvas_w + cache.chunk_px - 1) // cache.chunk_px
    old_n_cy = (canvas_h + cache.chunk_px - 1) // cache.chunk_px

    total_bytes = 0
    n_chunks = 0
    for cy in range(old_n_cy):
        for cx in range(old_n_cx):
            x0, y0 = cx * cache.chunk_px, cy * cache.chunk_px
            x1, y1 = min(x0 + cache.chunk_px, canvas_w), min(y0 + cache.chunk_px, canvas_h)
            total_bytes += (x1 - x0) * (y1 - y0) * 3
            n_chunks += 1
    assert n_chunks == old_n_cx * old_n_cy
    assert total_bytes == cache.max_bytes


def test_byte_budget_evicts_lru(cache) -> None:
    canvas_w, canvas_h = cache.canvas_dims()
    n_cx = (canvas_w + cache.chunk_px - 1) // cache.chunk_px
    n_cy = (canvas_h + cache.chunk_px - 1) // cache.chunk_px
    if n_cx * n_cy < 2:
        pytest.skip(f"fixture has only {n_cx * n_cy} chunk(s) at chunk_px={cache.chunk_px}, need >= 2")

    first = cache.get_chunk(0, 0, 0)
    # Shrink the budget to just under two chunks so the second insertion
    # must evict the first.
    cache.max_bytes = first.nbytes + first.nbytes // 2
    cache.get_chunk(0, 1 if n_cx > 1 else 0, 0 if n_cx > 1 else 1)

    assert (0, 0, 0) not in cache._cache
    assert len(cache._cache) == 1
    assert cache._cache_bytes == sum(c.nbytes for c in cache._cache.values())


@pytest.mark.parametrize("builder", _BUILDERS, ids=_BUILDER_IDS)
def test_explicit_max_chunks_is_count_only(builder) -> None:
    """Direct in-file mirror of tools/verify_iso_chunks.py:240 and
    tests/test_flat_chunks.py:231's own max_chunks=2 eviction tests -- a
    regression here is diagnosed by this file first."""
    cache = builder(max_chunks=2)
    assert cache.max_bytes is None
    assert cache.max_chunks == 2
    canvas_w, canvas_h = cache.canvas_dims()
    n_cx = (canvas_w + cache.chunk_px - 1) // cache.chunk_px
    n_cy = (canvas_h + cache.chunk_px - 1) // cache.chunk_px
    if n_cx * n_cy < 3:
        pytest.skip(f"fixture has only {n_cx * n_cy} chunk(s) at chunk_px={cache.chunk_px}, need >= 3")

    keys = [(cx, cy) for cy in range(n_cy) for cx in range(n_cx)][:3]
    for cx, cy in keys:
        cache.get_chunk(0, cx, cy)
    assert len(cache._cache) == 2
    assert (0, *keys[0]) not in cache._cache
    assert cache._cache_bytes == sum(c.nbytes for c in cache._cache.values())


@pytest.mark.parametrize("builder", _BUILDERS, ids=_BUILDER_IDS)
def test_explicit_large_max_chunks_never_gains_a_byte_bound(builder) -> None:
    """Direct in-file mirror of tools/verify_iso_chunks.py:327 and
    tests/test_flat_chunks.py:300's own max_chunks=100000 pattern -- both
    rely on an explicit large max_chunks NEVER thrashing under a byte
    bound, since they warm the whole canvas before calling
    invalidate_region() and expect the eviction that call causes to be the
    only eviction that happens."""
    cache = builder(max_chunks=100000)
    assert cache.max_bytes is None
    assert cache.max_chunks == 100000
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk
    assert cache._cache_bytes == sum(c.nbytes for c in cache._cache.values())
    # Warming the whole canvas would have overflowed almost any real byte
    # bound; nothing should have been evicted.
    n_cx = (canvas_w + cache.chunk_px - 1) // cache.chunk_px
    n_cy = (canvas_h + cache.chunk_px - 1) // cache.chunk_px
    assert len(cache._cache) == n_cx * n_cy


def test_cache_bytes_tracking_stays_consistent(cache) -> None:
    """The highest-value test in this file: _cache_bytes must equal the
    real sum of cached chunks' nbytes after every kind of mutation this
    phase touches, in sequence. The invalidate_region() decrement is the
    single easiest line in the whole phase to forget -- forgetting it
    silently under-evicts (the byte accounting drifts high) forever,
    without ever failing a shape/pixel assertion."""

    def assert_consistent():
        assert cache._cache_bytes == sum(c.nbytes for c in cache._cache.values())

    canvas_w, canvas_h = cache.canvas_dims()
    assert_consistent()  # nothing cached yet

    cache.get_chunk(0, 0, 0)
    assert_consistent()

    n_cx = (canvas_w + cache.chunk_px - 1) // cache.chunk_px
    n_cy = (canvas_h + cache.chunk_px - 1) // cache.chunk_px
    if n_cx * n_cy > 1:
        cache.get_chunk(0, n_cx - 1, n_cy - 1)
        assert_consistent()

    # invalidate_region over the whole canvas -- must evict everything AND
    # decrement _cache_bytes for every eviction.
    cache.invalidate_region((0, 0, canvas_w, canvas_h))
    assert len(cache._cache) == 0
    assert_consistent()

    # patch() writes in place -- nbytes of a cached chunk never changes,
    # so re-warm then patch and confirm the total is still exact.
    cache.render_rect(0, 0, canvas_w, canvas_h)
    assert_consistent()
    small_bbox = (0, 0, min(cache.chunk_px, canvas_w), min(cache.chunk_px, canvas_h))
    cache.patch(small_bbox)
    assert_consistent()
