"""patch() and patch_composite_rects() with a chunk pool (TASK-031.67): the
slices of cached chunks composite on worker threads through RectJob, and the
result must be the serial loop's: the same pixels, every cached chunk the same
array object, the same LRU order.

As in test_pooled_render_rect, "jobs were submitted" proves nothing about a
pooled kernel (the join may run them all inline), so the tests that claim a
worker ran one record the thread of every run() and require one off the
calling thread. Most tests drop POOL_MIN_PATCH_TILE_AREA to 0 so the fixture
map's patches pool; the gate test keeps the real value."""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_farm_terrain import FAKE_FARM_CONST, fake_farm  # noqa: F401 -- fake_farm is a fixture
from test_native_composite import (
    EDIT_X,
    EDIT_Y,
    FARM_X,
    FARM_Y,
    NEW_TERRAIN_ID,
    PLAIN_UNIT_CONST,
    native_kernel,  # noqa: F401 -- a fixture
    synthetic_textures,  # noqa: F401 -- a fixture
)
from test_pooled_render_rect import (
    WORKERS,
    _assert_same,
    _canvas,
    _composite_counts,
    _CountingExecutor,
    _FinishedAtSubmit,
    _make,
    _NeverStarts,
    _on_a_worker,
    _scenario_for,
    pool,  # noqa: F401 -- a fixture
)
from test_sprite_edit_bbox import sprite_install  # noqa: F401
from test_terrain_units_viewer import (  # noqa: F401 -- fixtures
    blend_install,
    blended_trees,
    tree_sprites,
)

from descape import composite_backend, debug_log, iso_geometry, native_composite, perf_trace, render, render_cache
from descape.render_cache import FlatChunkCache, RectJob, UnitSplice

import conftest

MIPS = (0, -1)


@pytest.fixture
def slice_threads(monkeypatch):
    """(key, thread ident) for every RectJob.run(), inline or on a worker."""
    import threading

    ran: list[tuple[tuple[int, int, int], int]] = []
    real = RectJob.run

    def run(self):
        ran.append((self.key, threading.get_ident()))
        return real(self)

    monkeypatch.setattr(RectJob, "run", run)
    return ran


def _pooled(cache, executor) -> None:
    cache.set_chunk_executor(executor, WORKERS)
    cache.POOL_MIN_PATCH_TILE_AREA = 0


def _warm(cache, mips=MIPS) -> None:
    for mip in mips:
        _canvas(cache, mip)


def _move_units(scenario, caches) -> list[tuple[int, int]]:
    """The fixture's mark and farm moved through invalidate_units() on every
    cache; returns the tiles they left and reached."""
    mm = scenario.map_manager
    mark = next(u for u in scenario.unit_manager.units[1] if u.unit_const == PLAIN_UNIT_CONST)
    farm = next(u for u in scenario.unit_manager.units[2] if u.unit_const == FAKE_FARM_CONST)
    splices, tiles = [], []
    for unit, player_id, (nx, ny) in (
        (mark, 1, (EDIT_X - 12.25, EDIT_Y + 7.75)),
        (farm, 2, (FARM_X + 4.5, FARM_Y - 3.5)),
    ):
        index = scenario.unit_manager.units[player_id].index(unit)
        old_tiles = tuple(render.unit_occupied_tiles(unit, mm.map_width, mm.map_height))
        old_own = (int(unit.x), int(unit.y))
        unit.x, unit.y = nx, ny
        new_tiles = tuple(render.unit_occupied_tiles(unit, mm.map_width, mm.map_height))
        splices.append(UnitSplice(player_id, index, unit, old_own, (int(nx), int(ny)), old_tiles, new_tiles))
        tiles += [*old_tiles, *new_tiles]
    scenario.unit_gen += 1
    for cache in caches:
        assert cache.can_splice(splices)
        cache.invalidate_units(splices)
    return tiles


def _paint_band(scenario) -> list[tuple[int, int]]:
    """A two-row band of a terrain id the map never had, across several chunks."""
    mm = scenario.map_manager
    tiles = [(EDIT_X - 30 + i, EDIT_Y + j) for i in range(60) for j in (0, 1)]
    for x, y in tiles:
        mm.terrain[y * mm.map_width + x].terrain_id = NEW_TERRAIN_ID
    return tiles


def _dirty_bbox(cache, style: str, scenario, tiles, pad: int = 0) -> tuple[int, int, int, int]:
    """The viewer's dirty bbox of `tiles` (REFERENCE pixels), units and sprites
    in, padded by `pad`. Syncs this cache's elevations, as the viewer's does."""
    w = scenario.map_manager.map_width
    fn = render.dirty_screen_bbox_iso if style == "stepped" else render.dirty_screen_bbox_sloped
    bbox = fn(scenario, [y * w + x for x, y in tiles], cache.elevations, cache.proj, with_units=True, with_sprites=True)
    assert bbox is not None
    return bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad


def _assert_like_serial(pooled, serial, before: dict, what: str) -> None:
    """Every cached chunk equals the serial twin's, in the same LRU order, and
    is the array object it was before the patch (unless the patch evicted it)."""
    assert list(pooled._cache) == list(serial._cache), f"{what}: LRU order or keys differ from serial"
    assert any(chunk.any() for chunk in pooled._cache.values()), f"{what}: every chunk is black -- vacuous"
    for key, chunk in pooled._cache.items():
        if key in before:
            assert chunk is before[key], f"{what}: {key} is a new array"
        want = serial._cache[key]
        if not np.array_equal(chunk, want):
            ys, _xs = np.nonzero(np.any(chunk != want, axis=-1))
            pytest.fail(f"{what}: chunk {key} differs from serial in {ys.size} px")


def _assert_like_fresh(cache, style: str, scenario, what: str) -> None:
    fresh = _make(style, scenario)
    for mip in MIPS:
        _assert_same(_canvas(cache, mip), _canvas(fresh, mip), f"{what}: mip {mip} vs a fresh cache")


def _twins(style: str, pool_):
    scenario = _scenario_for(style)
    pooled, serial = _make(style, scenario), _make(style, scenario)
    executor = _CountingExecutor(pool_)
    _pooled(pooled, executor)
    for cache in (pooled, serial):
        _warm(cache)
    executor.jobs.clear()  # the warm's own ChunkJobs
    return scenario, pooled, serial, executor


# --- T1: pixels, chunk identity and LRU ------------------------------------------------


@pytest.mark.parametrize("form", ["rects", "patch"])
@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_pooled_patch_equals_its_serial_twin_and_a_fresh_cache(
    native_kernel, sprite_install, fake_farm, synthetic_textures, blend_install, pool, slice_threads, style, form,  # noqa: F811
):
    """A unit batch and a terrain paint, then the per-chunk rects of each
    level (patch_composite_rects(chunk_rects())) or one patch() over their
    union, on a pooled cache and a serial twin, sprites and blending on.
    Every painted job must install: refused installs would recomposite
    synchronously to the same pixels and hide a pool doing nothing."""
    with composite_backend.use_backend("native"):
        scenario, pooled, serial, executor = _twins(style, pool)
        assert render.blend_effective(pooled.layers), "blending is off -- the blend path goes unchecked"
        landed: list = []
        real_install = pooled.install_slice

        def install(job, painted):
            ok = real_install(job, painted)
            if ok:
                landed.append(job)
            return ok

        pooled.install_slice = install
        tiles = _move_units(scenario, (pooled, serial)) + _paint_band(scenario)
        groups = [tiles[i : i + 12] for i in range(0, len(tiles), 12)]
        rects = [_dirty_bbox(pooled, style, scenario, group, pad=8) for group in groups]
        union = rects[0]
        for rect in rects[1:]:
            union = render_cache._union_bbox(union, rect)
        before = dict(pooled._cache)
        lru = list(pooled._cache)
        for cache in (pooled, serial):
            if form == "rects":
                for mip in MIPS:
                    cache.patch_composite_rects(cache.chunk_rects(rects, mip), levels=(mip,), rebuild_levels=(mip,))
            else:
                cache.patch(union, elevation_changed=set())
        assert len(executor.jobs) >= 2, f"{len(executor.jobs)} jobs -- the patch did not pool"
        assert _on_a_worker(slice_threads), "no slice kernel ran on a worker"
        assert all(isinstance(job, RectJob) for job in executor.jobs)
        assert landed == executor.jobs, f"{len(landed)} of {len(executor.jobs)} jobs installed"
        assert list(pooled._cache) == lru, "the patch changed the LRU order"
        _assert_like_serial(pooled, serial, before, f"{style} {form}")
        _assert_like_fresh(pooled, style, scenario, f"{style} {form}")


# --- T2: evicting levels --------------------------------------------------------------------


def _build_counts(cache) -> list[int]:
    builds: list[int] = []
    real = cache._commit_level

    def spy(mip, *args, **kwargs):
        builds.append(mip)
        return real(mip, *args, **kwargs)

    cache._commit_level = spy
    return builds


@pytest.mark.parametrize("case", ["stale-outside", "deferred"])
@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_evicting_levels_evict_the_serial_keys_and_a_stale_visible_level_builds_once(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, slice_threads, style, case,  # noqa: F811
):
    """rebuild_levels=(0,). stale-outside: a wholesale refresh leaves both
    levels stale, so -1 evicts and 0 builds once inside the patch. deferred:
    an elevation edit defers -1's splice (_defers_patch), so -1 evicts."""
    with composite_backend.use_backend("native"):
        scenario, pooled, serial, executor = _twins(style, pool)
        mm = scenario.map_manager
        tiles = [(EDIT_X + dx, EDIT_Y + dy) for dx in range(-6, 7) for dy in range(-6, 7)]
        if case == "deferred":
            for x, y in tiles:
                tile = mm.terrain[y * mm.map_width + x]
                tile.elevation = max(0, tile.elevation - 1) if style == "sloped" else (tile.elevation + 3) % (
                    iso_geometry.MAX_ELEVATION + 1
                )
        else:
            _paint_band(scenario)
            tiles += [(EDIT_X - 30 + i, EDIT_Y) for i in range(60)]
        builds = {id(cache): _build_counts(cache) for cache in (pooled, serial)}
        before = dict(pooled._cache)
        for cache in (pooled, serial):
            changed: set = set()
            w = mm.map_width
            fn = render.dirty_screen_bbox_iso if style == "stepped" else render.dirty_screen_bbox_sloped
            bbox = fn(
                scenario, [y * w + x for x, y in tiles], cache.elevations, cache.proj, with_units=True,
                with_sprites=True, elevation_changed=changed,
            )
            if case == "deferred":
                assert changed, "no elevation moved -- vacuous"
                cache.patch(bbox, elevation_changed=changed, rebuild_levels=(0,))
                assert cache._defers_patch(-1), "level -1 did not defer -- vacuous"
            else:
                cache.patch(bbox, elevation_changed=None, rebuild_levels=(0,))
        assert executor.jobs, "the patch did not pool"
        assert _on_a_worker(slice_threads)
        evicted = set(before) - set(pooled._cache)
        assert evicted and all(key[0] == -1 for key in evicted), f"{case}: evicted {sorted(evicted)[:4]}"
        _assert_like_serial(pooled, serial, before, f"{style} {case}")
        if case == "stale-outside":
            assert builds[id(pooled)] == builds[id(serial)] == [0], f"builds {builds}"
        _assert_like_fresh(pooled, style, scenario, f"{style} {case}")


# --- T3: declines -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["pool", "worker", "inline"])
@pytest.mark.parametrize("how", ["prepare-none", "run-false", "run-raises"])
def test_a_declined_slice_composites_synchronously_exactly_once(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, monkeypatch, how, path,  # noqa: F811
):
    """`worker` and `inline` pin each of the join's two paths alone; `pool` is the live mix."""
    with composite_backend.use_backend("native"):
        scenario = _scenario_for("stepped")
        pooled, serial = _make("stepped", scenario), _make("stepped", scenario)
        executor = {"pool": lambda: pool, "worker": lambda: _FinishedAtSubmit(pool), "inline": _NeverStarts}[path]()
        _pooled(pooled, executor)
        for cache in (pooled, serial):
            _warm(cache, (0,))
        tiles = _paint_band(scenario)
        bbox = _dirty_bbox(pooled, "stepped", scenario, tiles, pad=40)
        keys = [t[:3] for t in pooled._patch_targets(bbox, levels=(0,))]
        assert len(keys) >= 6 and len(set(keys)) == len(keys)
        declined = {keys[0], keys[len(keys) // 2], keys[-1]}
        if how == "prepare-none":
            real_prepare = pooled.prepare_rect_job

            def prepare(mip, cx, cy, *rect, heights=None):
                return None if (mip, cx, cy) in declined else real_prepare(mip, cx, cy, *rect, heights=heights)

            pooled.prepare_rect_job = prepare
        else:
            real_run = RectJob.run

            def run(self):
                if self.key in declined:
                    if how == "run-raises":
                        raise RuntimeError(f"injected kernel failure {self.key}")
                    return False
                return real_run(self)

            monkeypatch.setattr(RectJob, "run", run)
        before = dict(pooled._cache)
        counts = _composite_counts(pooled)
        debug_log.clear()
        for cache in (pooled, serial):
            cache.patch(bbox, elevation_changed=set(), levels=(0,))
        _assert_like_serial(pooled, serial, before, how)
        assert counts == dict.fromkeys(declined, 1), f"{how}: synchronous composites {counts}"
        if how == "run-raises":
            log = debug_log.get_log_text()
            assert all(f"chunk {key} raised" in log for key in declined), log


def test_a_raising_prepare_leaves_no_unpatched_target_cached(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool,  # noqa: F811
):
    """The raise propagates, no kernel stays queued, and every target not yet
    installed is evicted, so the next paint composites it fresh."""
    with composite_backend.use_backend("native"):
        scenario, pooled, _serial, executor = _twins("stepped", pool)
        tiles = _paint_band(scenario)
        bbox = _dirty_bbox(pooled, "stepped", scenario, tiles, pad=40)
        keys = [t[:3] for t in pooled._patch_targets(bbox, levels=(0,))]
        assert len(keys) >= 4
        real_prepare = pooled.prepare_rect_job

        def prepare(mip, cx, cy, *rect, heights=None):
            if (mip, cx, cy) == keys[2]:
                raise RuntimeError("injected prepare failure")
            return real_prepare(mip, cx, cy, *rect, heights=heights)

        pooled.prepare_rect_job = prepare
        with pytest.raises(RuntimeError, match="injected prepare failure"):
            pooled.patch(bbox, elevation_changed=set(), levels=(0,))
        assert len(executor.jobs) == 2
        assert not any(key in pooled._cache for key in keys), "an unpatched target stayed cached"
        assert pooled._cache_bytes == sum(c.nbytes for c in pooled._cache.values())
        pooled.prepare_rect_job = real_prepare
        _assert_like_fresh(pooled, "stepped", scenario, "after a raising prepare")


# --- T4: install_slice() --------------------------------------------------------------------


def test_install_slice_refuses_a_stale_epoch_an_evicted_chunk_and_a_replaced_one(
    native_kernel, sprite_install, fake_farm, synthetic_textures,  # noqa: F811
):
    with composite_backend.use_backend("native"):
        cache = _make("stepped", _scenario_for("stepped"))
        _warm(cache, (0,))
        key = (0, *_chunk_with_band(cache))
        _mip, cx, cy = key
        x0, y0 = cx * cache.chunk_px, cy * cache.chunk_px
        rect = (x0 + 10, y0 + 20, x0 + 200, y0 + 150)
        lru = list(cache._cache)

        def painted_job():
            job = cache.prepare_rect_job(*key, *rect)
            assert job is not None and job.chunk is cache._cache[key] and job.slice == rect
            assert job.run()
            _assert_same(job.scratch, cache._composite_rect(0, *rect), "a slice job's pixels")
            job.scratch[:] = 255  # so a write shows
            return job

        def written(chunk) -> bool:
            return bool((chunk[20:150, 10:200] == 255).all())

        job = painted_job()
        assert not cache.install_slice(job, False), "a declined kernel installed"
        job = painted_job()
        cache._mutation_epoch += 1
        assert not cache.install_slice(job, True), "a stale epoch installed"
        assert not written(job.chunk)

        job = painted_job()
        cache.invalidate_region((x0, y0, x0 + 1, y0 + 1), levels=(0,))
        assert key not in cache._cache
        assert not cache.install_slice(job, True), "a slice of an evicted chunk installed"
        assert key not in cache._cache, "install_slice() inserted a chunk"
        assert not written(job.chunk)

        cache.get_chunk(*key)
        job = painted_job()
        replacement = cache._cache[key].copy()
        cache._cache[key] = replacement
        assert not cache.install_slice(job, True), "a slice of a replaced chunk installed"
        assert not written(job.chunk) and not written(replacement)

        job = painted_job()
        order = list(cache._cache)
        assert cache.install_slice(job, True), "control: a current job must install"
        assert written(cache._cache[key]) and cache._cache[key] is job.chunk
        assert list(cache._cache) == order, "an install moved the LRU order"
        assert lru, "vacuous"

        cache.invalidate_region((x0, y0, x0 + 1, y0 + 1), levels=(0,))
        assert cache.prepare_rect_job(*key, *rect) is None, "a slice job for an uncached chunk"


# --- T5: overlapping rects ----------------------------------------------------------------


def test_overlapping_rects_in_one_chunk_equal_serial_and_install_in_walk_order(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, slice_threads,  # noqa: F811
):
    """Overlapping slices paint the same pixels, so the order is pinned
    directly: installs run in walk order, the serial loop's write order."""
    with composite_backend.use_backend("native"):
        scenario, pooled, serial, executor = _twins("stepped", pool)
        _paint_band(scenario)
        cx, cy = _chunk_with_band(pooled)
        x0, y0 = cx * pooled.chunk_px, cy * pooled.chunk_px
        rects = [
            (x0 + 50, y0 + 50, x0 + 300, y0 + 300),
            (x0 + 200, y0 + 200, x0 + 450, y0 + 450),
            (x0 + 100, y0 + 10, x0 + 500, y0 + 260),
        ]
        installed: list = []
        real = pooled.install_slice

        def install(job, painted):
            ok = real(job, painted)
            installed.append(job.slice if ok else None)
            return ok

        pooled.install_slice = install
        before = dict(pooled._cache)
        for cache in (pooled, serial):
            cache.patch_composite_rects(rects, levels=(0,), rebuild_levels=(0,))
        assert len(executor.jobs) == len(rects)
        assert installed == rects, f"installs out of walk order: {installed}"
        _assert_like_serial(pooled, serial, before, "overlapping rects")


def _chunk_with_band(cache) -> tuple[int, int]:
    """A mip-0 chunk under the painted band's middle."""
    sx, sy = iso_geometry.tile_screen_origin(EDIT_X, EDIT_Y, int(cache.elevations[EDIT_Y, EDIT_X]), cache.proj)
    return int(sx) // cache.chunk_px, int(sy) // cache.chunk_px


# --- T6: the gate -----------------------------------------------------------------------------


def test_pooling_engages_only_where_it_can(native_kernel, sprite_install, fake_farm, synthetic_textures, pool):  # noqa: F811
    with composite_backend.use_backend("native"):
        scenario = _scenario_for("stepped")
        cache = _make("stepped", scenario)
        executor = _CountingExecutor(pool)
        cache.set_chunk_executor(executor, WORKERS)
        assert cache.POOL_MIN_PATCH_TILE_AREA == 1100
        _warm(cache)
        c = cache.chunk_px
        cx, cy = 3, 2

        def submits(bbox, levels=(0,)) -> int:
            executor.jobs.clear()
            cache.patch(bbox, elevation_changed=set(), levels=levels)
            return len(executor.jobs)

        def tiles(bbox, levels=(0,)) -> tuple[int, float]:
            targets = list(cache._patch_targets(bbox, levels=levels))
            return len(targets), sum(
                (t[6] - t[4]) * (t[7] - t[5]) / cache.mip_tile_px(t[0]) ** 2 for t in targets
            )

        whole = (cx * c, cy * c, (cx + 1) * c, (cy + 1) * c)
        assert tiles(whole)[0] == 1
        assert submits(whole) == 0, "one slice was pooled"

        small = (cx * c - 100, cy * c - 100, cx * c + 100, cy * c + 100)
        n, area = tiles(small)
        assert n >= 2 and area < cache.POOL_MIN_PATCH_TILE_AREA
        assert submits(small) == 0, "a patch under the tile-area gate was pooled"

        # A brush-9 Draw step at mip 0, both levels resident, at a chunk corner.
        mm = scenario.map_manager
        brush = [(EDIT_X + dx, EDIT_Y + dy) for dx in range(-4, 5) for dy in range(-4, 5)]
        w = mm.map_width
        draw = render.dirty_screen_bbox_iso(
            scenario, [y * w + x for x, y in brush], cache.elevations, cache.proj, with_units=True, with_sprites=True
        )
        n, area = tiles(draw, levels=None)
        assert n >= 2 and area < cache.POOL_MIN_PATCH_TILE_AREA, (n, area)
        assert submits(draw, levels=None) == 0, "a brush-9 Draw step was pooled"

        level = (0, 0, *cache.canvas_dims(0))
        n, area = tiles(level, levels=(-1,))
        assert area >= cache.POOL_MIN_PATCH_TILE_AREA
        assert submits(level, levels=(-1,)) >= n // 2, "control: a patch past the gate must pool"

        cache.POOL_MIN_PATCH_TILE_AREA = 0
        with composite_backend.use_backend("numpy"):
            assert submits(small) == 0, "the numpy backend pooled"
        cache.set_chunk_executor(None, 0)
        assert submits(small) == 0, "no executor still pooled"

        flat = FlatChunkCache(scenario, render.tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True)
        _pooled(flat, executor)
        _canvas(flat, 0)
        executor.jobs.clear()
        prepared = []
        flat.prepare_rect_job = lambda *a, **k: prepared.append(a)
        flat.patch_rects([(0, 0, 400, 400), (600, 600, 1400, 1100)])
        assert prepared == [] and executor.jobs == [], "Flat pooled"


# --- T7: batches -----------------------------------------------------------------------------


def test_a_two_slice_batch_gives_the_same_pixels_and_joins_every_batch(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, slice_threads, monkeypatch,  # noqa: F811
):
    with composite_backend.use_backend("native"):
        scenario, pooled, serial, executor = _twins("stepped", pool)
        pooled.POOL_BATCH = 2
        joins: list[int] = []
        real_join = render_cache.join_chunk_jobs
        monkeypatch.setattr(render_cache, "join_chunk_jobs", lambda s: (joins.append(len(s)), real_join(s))[1])
        landed: list = []
        real_install = pooled.install_slice

        def install(job, painted):
            ok = real_install(job, painted)
            if ok:
                landed.append(job)
            return ok

        pooled.install_slice = install
        tiles = _paint_band(scenario)
        bbox = _dirty_bbox(pooled, "stepped", scenario, tiles, pad=40)
        n = len(list(pooled._patch_targets(bbox)))
        assert n >= 5
        before = dict(pooled._cache)
        for cache in (pooled, serial):
            cache.patch(bbox, elevation_changed=set())
        assert len(joins) == math.ceil(n / 2) and sum(joins) == len(executor.jobs) == n, joins
        assert landed == executor.jobs, f"{len(landed)} of {len(executor.jobs)} jobs installed"
        assert _on_a_worker(slice_threads)
        _assert_like_serial(pooled, serial, before, "POOL_BATCH 2")


# --- T8: the unit pack -------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["splice", "rebuild", "rebuild-small"])
@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_no_prepare_writes_the_pack_after_the_first_submit(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, monkeypatch, style, case,  # noqa: F811
):
    """splice: a unit batch's refreshed pack, its moved tiles pending.
    rebuild: a wholesale refresh, so the level and its pack are rebuilt inside
    the patch, every unit tile pending. rebuild-small: the same with two tiny
    slices, which derive their candidates only. Each way the batch derives them
    before the first prepare, so no prepare writes the pack while a job may
    read it and no copy-on-write copy is stranded in a job."""
    with composite_backend.use_backend("native"):
        scenario = _scenario_for(style)
        cache = _make(style, scenario)
        _warm(cache, (-1,))
        executor = _CountingExecutor(pool)
        _pooled(cache, executor)
        if case == "splice":
            _move_units(scenario, (cache,))
            pack = cache._levels[-1].unit_pack
            assert pack is not None and pack.pending().size, "nothing pending -- vacuous"
        copies, writes = [], []
        real_unshare, real_write = native_composite.UnitPack._unshare, native_composite.UnitPack._write_tiles

        def unshare(self):
            if self._shared:
                copies.append(len(executor.jobs))
            real_unshare(self)

        def write_tiles(self, tiles):
            tiles = list(tiles)
            if tiles:
                writes.append(len(executor.jobs))
            real_write(self, tiles)

        monkeypatch.setattr(native_composite.UnitPack, "_unshare", unshare)
        monkeypatch.setattr(native_composite.UnitPack, "_write_tiles", write_tiles)
        if case == "rebuild-small":
            # Two 8 px rects on unit tiles: far more pending than slice area, so the candidates branch.
            cache._refresh_all_units()
            rects = []
            for x, y in ((EDIT_X, EDIT_Y), (FARM_X, FARM_Y)):
                sx, sy = iso_geometry.tile_screen_origin(x, y, int(cache.elevations[y, x]), cache.proj)
                sx, sy = int(sx) + cache.tile_px // 2, int(sy) + cache.tile_px // 4
                rects.append((sx - 4, sy - 4, sx + 4, sy + 4))
            cache.patch_composite_rects(rects, levels=(-1,), rebuild_levels=(-1,))
        else:
            whole = (0, 0, *cache.canvas_dims(0))
            cache.patch(whole, elevation_changed=None if case == "rebuild" else set(), levels=(-1,))
        jobs = executor.jobs
        assert len(jobs) >= 2
        assert writes and set(writes) == {0}, f"a prepare derived pack tiles after the first submit: {writes}"
        assert [n for n in copies if n > 0] == [], f"copy-on-write copies after the first submit: {copies}"
        pack = cache._levels[-1].unit_pack
        assert {id(job.holds[0]) for job in jobs} == {id(pack)}
        if case == "rebuild-small":
            assert pack.pending().size, "every pending tile was derived -- the candidates branch never ran"
        _assert_like_fresh(cache, style, scenario, f"{style} {case}")


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_the_slice_pre_derive_takes_pending_tiles_or_the_slices_candidates(
    native_kernel, sprite_install, fake_farm, synthetic_textures, style,  # noqa: F811
):
    """Few pending tiles against the slices' area: every pending tile is
    derived. More pending than that: only the slices' candidates are."""
    with composite_backend.use_backend("native"):
        scenario = _scenario_for(style)
        cache = _make(style, scenario)
        lvl = cache._level(0)
        pack = cache._native_units(lvl, 0)
        pending = pack.pending()
        assert pending.size >= 2
        w = scenario.map_manager.map_width
        x, y = int(pending[0] % w), int(pending[0] // w)
        sx, sy = iso_geometry.tile_screen_origin(x, y, int(cache.elevations[y, x]), cache.proj)
        tiny = (int(sx), int(sy), int(sx) + 2, int(sy) + 2)
        cache._derive_pack_for_slices(0, [tiny])
        candidates = render._rect_candidates(
            scenario, *tiny, lvl.proj, cache.with_units, lvl.building_bboxes, lvl.bystander_grid
        )
        idx = candidates[:, 1].astype(np.int64) * w + candidates[:, 0]
        assert pack.ready[idx].all(), "a slice's candidates were left pending"
        assert pack.pending().size, "every pending tile was derived for a 4 px slice"
        cache._derive_pack_for_slices(0, [(0, 0, *cache.canvas_dims(0))])
        assert pack.pending().size == 0, "a whole-level slice left pending tiles"


# --- T9: Sloped pick planes --------------------------------------------------------------------


def test_a_pooled_sloped_patch_rewrites_a_partly_covered_pick_plane(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, slice_threads,  # noqa: F811
):
    with composite_backend.use_backend("native"):
        scenario = _scenario_for("sloped")
        cache = _make("sloped", scenario)
        _warm(cache, (0,))
        executor = _CountingExecutor(pool)
        _pooled(cache, executor)
        mm = scenario.map_manager
        tiles = [(EDIT_X + dx, EDIT_Y + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
        for x, y in tiles:
            tile = mm.terrain[y * mm.map_width + x]
            tile.elevation = max(0, tile.elevation - 1)
        changed: set = set()
        w = mm.map_width
        bbox = render.dirty_screen_bbox_sloped(
            scenario, [y * w + x for x, y in tiles], cache.elevations, cache.proj, with_units=True,
            with_sprites=True, elevation_changed=changed,
        )
        c = cache.chunk_px
        partial = []
        for cy in range(bbox[1] // c, (bbox[3] - 1) // c + 1):
            for cx in range(bbox[0] // c, (bbox[2] - 1) // c + 1):
                plane = cache._pick_plane(cx, cy)
                ix = min(bbox[2], cx * c + plane.shape[1]) - max(bbox[0], cx * c)
                iy = min(bbox[3], cy * c + plane.shape[0]) - max(bbox[1], cy * c)
                if ix * iy / plane.size <= render_cache.PICK_PLANE_PATCH_MAX_FRACTION:
                    partial.append((cx, cy))
        # The plane memo holds MAX_PICK_PLANES; only those still resident are rewritten.
        partial = [key for key in partial if key in cache._pick_planes]
        assert partial, "no plane is partly covered -- vacuous"
        cache.patch(bbox, elevation_changed=changed)
        # No worker assert: with a few jobs the join may rightly run them all inline.
        assert len(executor.jobs) >= 2
        fresh = _make("sloped", scenario)
        for key in partial:
            assert key in cache._pick_planes, f"plane {key} was dropped, not rewritten"
            assert np.array_equal(cache._pick_planes[key], fresh._pick_plane(*key)), f"plane {key} is stale"
        _assert_same(_canvas(cache, 0), _canvas(fresh, 0), "sloped colour after a pooled patch")


# --- T10: the viewer --------------------------------------------------------------------------


@pytest.fixture
def traced(monkeypatch):
    """Perf Trace on, with fresh module state restored afterwards."""
    debug_log.clear()
    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "_idle_scheduler", None)
    monkeypatch.setattr(perf_trace, "_sprite_counter", lambda: (0, 0))
    monkeypatch.setattr(perf_trace, "_enabled", True)
    yield
    debug_log.clear()


@pytest.mark.parametrize("pooled", [True, False], ids=["pooled", "serial"])
def test_a_pooled_patch_says_so_on_its_perf_line(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, traced, pooled,  # noqa: F811
):
    with composite_backend.use_backend("native"):
        scenario, cache, _serial, executor = _twins("stepped", pool)
        if not pooled:
            cache.set_chunk_executor(None, 0)
        tiles = _paint_band(scenario)
        bbox = _dirty_bbox(cache, "stepped", scenario, tiles, pad=40)
        n = len(list(cache._patch_targets(bbox)))
        perf_trace.flush_pending_op()
        debug_log.clear()
        with perf_trace.op("Undo"), perf_trace.phase("unit_patch"):
            cache.patch(bbox, elevation_changed=set())
        perf_trace.flush_pending_op()
    line = next(line for line in debug_log.get_log_text().splitlines() if "perf op undo" in line)
    if pooled:
        assert f"patch_pool={len(executor.jobs)}/{n}" in line, line
    else:
        assert "patch_pool=" not in line, line


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize(("style", "repaint"), [("Stepped", "rects"), ("Sloped", "patch")])
def test_an_over_cap_batch_its_undo_and_redo_pool_their_patch_and_match_a_fresh_render(
    style, repaint, blended_trees, traced, monkeypatch  # noqa: F811
) -> None:
    """The viewer's over-cap batch repaint (_repaint_unit_edit_split()), with
    the window's own pool: Stepped's per-chunk rect branch, Sloped's patch
    branch. Every step pools its slices, says so on its perf line, and leaves
    every oracle level equal to a fresh build."""
    import test_terrain_units_viewer as tuv

    window = tuv._blended_window(style, monkeypatch, repaint)
    try:
        cache = window._cache
        assert cache._chunk_executor is not None, "the window has no chunk pool"
        executor = _CountingExecutor(cache._chunk_executor)
        cache.set_chunk_executor(executor, WORKERS)
        cache.POOL_MIN_PATCH_TILE_AREA = 0
        rect_patches = tuv._spy_rect_patches(window, monkeypatch)

        def each(step, act):
            executor.jobs.clear()
            act()
            assert window._cache is cache, f"{step}: the cache was replaced"
            assert any(isinstance(job, RectJob) for job in executor.jobs), f"{step}: no slice pooled"
            tuv._check_against_fresh(window, step)

        if style == "Stepped":
            tuv._long_batches(window, each)
            assert len(rect_patches) == len(tuv.LONG_STEPS), rect_patches
        else:
            tuv._draw(window, tuv.FOREST_PALM, tuv.EARLIER_TILES)
            acts = {"stroke": lambda: tuv._draw(window, tuv.FOREST_OAK, tuv.STROKE_TILES), "undo": window.undo,
                    "redo": window.redo}
            for step in tuv.BLENDED_STEPS:
                tuv._render_levels(window)
                each(step, acts[step])
            assert rect_patches == []
    finally:
        window.edit_history.mark_saved()
        window.close()
