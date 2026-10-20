"""render_rect() with a chunk pool (TASK-031.64): the misses composite on
worker threads through the ChunkJob path, and the result must be the serial
loop's, pixels and cache state alike.

"Jobs were submitted" proves nothing about a pooled kernel (the join may run
them all inline), so the tests that claim a worker ran one record the thread
of every ChunkJob.run() and require at least one off the calling thread."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pytest
from test_farm_terrain import fake_farm  # noqa: F401 -- a fixture
from test_native_composite import (
    EDIT_X,
    EDIT_Y,
    _chunk_of_tile,
    _cliff_scenario,
    _edit_cache,
    _patch_tiles,
    _sloped_scenario,
    native_kernel,  # noqa: F401 -- a fixture
    synthetic_textures,  # noqa: F401 -- a fixture
)
from test_sprite_edit_bbox import sprite_install  # noqa: F401
from test_terrain_blend_render import _blend_cache, _scenario, blend_install  # noqa: F401

from descape import composite_backend, debug_log, iso_geometry, native_composite, perf_trace, render
from descape.grid_overlay import grid_bake
from descape.render_cache import ChunkJob, FlatChunkCache, IsoChunkCache, SlopedChunkCache

MIPS = (0, -1)
WORKERS = 3


@pytest.fixture
def pool():
    executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="test-pooled-paint")
    yield executor
    executor.shutdown(wait=True)


@pytest.fixture
def kernel_threads(monkeypatch):
    """(key, thread ident) for every ChunkJob.run(), inline or on a worker."""
    ran: list[tuple[tuple[int, int, int], int]] = []
    real = ChunkJob.run

    def run(self):
        ran.append((self.key, threading.get_ident()))
        return real(self)

    monkeypatch.setattr(ChunkJob, "run", run)
    return ran


class _CountingExecutor:
    """Wraps a real pool and keeps every job it was handed."""

    def __init__(self, inner):
        self.inner = inner
        self.jobs: list[ChunkJob] = []

    def submit(self, fn):
        self.jobs.append(fn.__self__)
        return self.inner.submit(fn)


def _scenario_for(style: str):
    return _cliff_scenario() if style == "stepped" else _sloped_scenario()


def _make(style: str, scenario, max_chunks: int | None = None):
    """_edit_cache() with an optional count cap (sprites and grid on)."""
    if max_chunks is None:
        return _edit_cache(style, scenario, sprites=True)
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = render.elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, max_chunks=max_chunks, sprites=True)
    else:
        elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, max_chunks=max_chunks, sprites=True)
    cache.set_grid(grid_bake(True, 60, 2))
    return cache


def _canvas(cache, mip: int) -> np.ndarray:
    return cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)


def _assert_same(got: np.ndarray, want: np.ndarray, what: str) -> None:
    assert want.any(), f"{what}: rendered all black -- proves nothing"
    assert got.shape == want.shape, what
    if not np.array_equal(got, want):
        ys, xs = np.nonzero(np.any(got != want, axis=-1))
        pytest.fail(f"{what}: {ys.size} px differ, first at (y={ys[0]}, x={xs[0]})")


def _on_a_worker(ran) -> bool:
    main = threading.get_ident()
    return any(ident != main for _key, ident in ran)


def _composite_counts(cache) -> dict:
    """Spies cache._composite_rect: chunk key -> synchronous composites."""
    counts: dict = {}
    real = cache._composite_rect

    def spy(mip, x0, y0, x1, y1):
        key = (mip, x0 // cache.chunk_px, y0 // cache.chunk_px)
        counts[key] = counts.get(key, 0) + 1
        return real(mip, x0, y0, x1, y1)

    cache._composite_rect = spy
    return counts


def _landings(cache) -> list:
    """Spies cache.install_chunk: the keys of every job that landed."""
    landed: list = []
    real = cache.install_chunk

    def spy(job, painted):
        ok = real(job, painted)
        if ok:
            landed.append(job.key)
        return ok

    cache.install_chunk = spy
    return landed


# --- T1, T6: pixels ---------------------------------------------------------


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_pooled_canvas_equals_a_serial_one_over_an_evict_and_repaint_loop(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, kernel_threads, style,  # noqa: F811
):
    """Every job's result is the one painted: each submitted job lands."""
    scenario = _scenario_for(style)
    with composite_backend.use_backend("native"):
        pooled, serial = _make(style, scenario), _make(style, scenario)
        executor = _CountingExecutor(pool)
        pooled.set_chunk_executor(executor, WORKERS)
        landed = _landings(pooled)
        w, h = pooled.canvas_dims(0)
        evictions = [None, (0, 0, w // 2, h // 2), (w // 3, h // 4, w, h), (0, 0, w, h)]
        for round_, bbox in enumerate(evictions):
            if bbox is not None:
                pooled.invalidate_region(bbox)
                serial.invalidate_region(bbox)
            for mip in MIPS:
                kernel_threads.clear()
                executor.jobs.clear()
                landed.clear()
                _assert_same(_canvas(pooled, mip), _canvas(serial, mip), f"{style} round {round_} mip {mip}")
                assert _on_a_worker(kernel_threads), f"{style} round {round_} mip {mip}: no kernel ran on a worker"
                assert executor.jobs and landed == [job.key for job in executor.jobs], (
                    f"{style} round {round_} mip {mip}: {len(landed)} of {len(executor.jobs)} jobs landed"
                )
            assert list(pooled._cache) == list(serial._cache), f"{style} round {round_}: LRU order differs"


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_after_an_elevation_edit_a_pooled_repaint_equals_a_fresh_serial_cache(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, kernel_threads, style,  # noqa: F811
):
    scenario = _scenario_for(style)
    with composite_backend.use_backend("native"):
        cache = _make(style, scenario)
        cache.set_chunk_executor(pool, WORKERS)
        for mip in MIPS:
            _canvas(cache, mip)
        mm = scenario.map_manager
        tiles = [(EDIT_X + dx, EDIT_Y + dy) for dx in range(-2, 3) for dy in range(-2, 3)]
        for x, y in tiles:
            tile = mm.terrain[y * mm.map_width + x]
            tile.elevation = max(0, tile.elevation - 1) if style == "sloped" else (tile.elevation + 5) % (
                iso_geometry.MAX_ELEVATION + 1
            )
        _patch_tiles(cache, style, scenario, tiles, True)
        cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
        fresh = _make(style, scenario)
        landed = _landings(cache)
        for mip in MIPS:
            kernel_threads.clear()
            landed.clear()
            _assert_same(_canvas(cache, mip), _canvas(fresh, mip), f"{style} mip {mip} after an elevation edit")
            assert _on_a_worker(kernel_threads), f"{style} mip {mip}: no kernel ran on a worker"
            assert landed, f"{style} mip {mip}: no pooled job landed"


# --- T2: blending ------------------------------------------------------------


@pytest.mark.parametrize("kind", ["stepped", "sloped"])
def test_a_pooled_blended_canvas_equals_the_numpy_backends(
    native_kernel, blend_install, pool, kernel_threads, kind,  # noqa: F811
):
    scenario = _scenario(kind)
    with composite_backend.use_backend("numpy"):
        reference = _blend_cache(kind, scenario)
        want = {mip: _canvas(reference, mip) for mip in MIPS}
    with composite_backend.use_backend("native"):
        cache = _blend_cache(kind, scenario)
        executor = _CountingExecutor(pool)
        cache.set_chunk_executor(executor, WORKERS)
        landed = _landings(cache)
        for mip in MIPS:
            kernel_threads.clear()
            executor.jobs.clear()
            landed.clear()
            _assert_same(_canvas(cache, mip), want[mip], f"blended {kind} mip {mip}")
            assert _on_a_worker(kernel_threads), f"blended {kind} mip {mip}: no kernel ran on a worker"
            assert executor.jobs and landed == [job.key for job in executor.jobs], f"blended {kind} mip {mip}"


# --- T3, T7: cache state -------------------------------------------------------


def _block(cache, mip: int, cols: int, rows: int) -> tuple[list, tuple[int, int, int, int]]:
    """A cols x rows block of whole chunks at the canvas centre: its keys in
    row-major order and its level pixel rect."""
    w, h = cache.canvas_dims(mip)
    c = cache.chunk_px
    cx0, cy0 = (w // c - cols) // 2, (h // c - rows) // 2
    assert cx0 >= 0 and cy0 >= 0, "canvas too small for the block"
    keys = [(mip, cx, cy) for cy in range(cy0, cy0 + rows) for cx in range(cx0, cx0 + cols)]
    return keys, (cx0 * c, cy0 * c, (cx0 + cols) * c, (cy0 + rows) * c)


@pytest.mark.parametrize("max_chunks", [None, 2], ids=["unbounded", "max2"])
@pytest.mark.parametrize("start", ["cold", "half-warm"])
def test_pooled_lru_order_equals_the_serial_loops(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, start, max_chunks,  # noqa: F811
):
    """The half-warm start caches the block's LAST keys, behind at least
    max_chunks misses, so at max2 every one of them is cached when the batch
    prepares and evicted by earlier installs before its turn: the serial loop
    composites it again there, and so must the pooled one."""
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        pooled, serial = _make("stepped", scenario, max_chunks), _make("stepped", scenario, max_chunks)
        executor = _CountingExecutor(pool)
        pooled.set_chunk_executor(executor, WORKERS)
        keys, rect = _block(pooled, 0, 3, 2)
        warm = keys[-2:] if start == "half-warm" else []
        for cache in (pooled, serial):
            cache._level(0)
            for key in warm:
                cache.get_chunk(*key)
        pooled_counts, serial_counts = _composite_counts(pooled), _composite_counts(serial)

        _assert_same(pooled.render_rect(*rect, mip=0), serial.render_rect(*rect, mip=0), f"{start} block")
        assert list(pooled._cache) == list(serial._cache), f"{start} max_chunks={max_chunks}: LRU order differs"
        assert pooled._cache_bytes == serial._cache_bytes
        assert sorted(job.key for job in executor.jobs) == sorted(k for k in keys if k not in warm), (
            "the batch must prepare exactly the keys uncached at its start"
        )
        for key in warm:
            want = serial_counts.get(key, 0)
            if max_chunks == 2:
                assert want == 1, f"{key}: the serial loop didn't recomposite an evicted warm key -- vacuous"
            assert pooled_counts.get(key, 0) == want, f"{key}: {pooled_counts.get(key, 0)} composites, serial {want}"


def test_a_two_key_batch_gives_the_same_pixels_and_lru(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, kernel_threads,  # noqa: F811
):
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        for max_chunks in (None, 5):
            pooled, serial = _make("stepped", scenario, max_chunks), _make("stepped", scenario, max_chunks)
            pooled.POOL_BATCH = 2
            pooled.set_chunk_executor(pool, WORKERS)
            for mip in MIPS:
                kernel_threads.clear()
                _assert_same(_canvas(pooled, mip), _canvas(serial, mip), f"POOL_BATCH 2 mip {mip}")
                assert _on_a_worker(kernel_threads), f"mip {mip}: no kernel ran on a worker"
                assert list(pooled._cache) == list(serial._cache), f"max_chunks={max_chunks} mip {mip}: LRU differs"


# --- T4: declines --------------------------------------------------------------


class _FinishedAtSubmit:
    """A real pool whose futures are done before submit returns: every job
    takes the join's worker-result path, none the inline one."""

    def __init__(self, inner):
        self.inner = inner

    def submit(self, fn):
        future = self.inner.submit(fn)
        wait([future])
        return future


@pytest.mark.parametrize("path", ["pool", "worker", "inline"])
@pytest.mark.parametrize("how", ["prepare-none", "run-false", "run-raises"])
def test_a_declined_job_composites_synchronously_exactly_once(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, monkeypatch, how, path,  # noqa: F811
):
    """`worker` and `inline` pin each of the join's two paths alone; `pool` is the live mix."""
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        pooled, serial = _make("stepped", scenario), _make("stepped", scenario)
        executor = {"pool": lambda: pool, "worker": lambda: _FinishedAtSubmit(pool), "inline": _NeverStarts}[path]()
        pooled.set_chunk_executor(executor, WORKERS)
        keys, rect = _block(pooled, 0, 4, 3)
        declined = {keys[0], keys[5], keys[-1]}
        if how == "prepare-none":
            real_prepare = pooled.prepare_chunk_job

            def prepare(mip, cx, cy, heights=None):
                return None if (mip, cx, cy) in declined else real_prepare(mip, cx, cy, heights=heights)

            pooled.prepare_chunk_job = prepare
        else:
            real_run = ChunkJob.run

            def run(self):
                if self.key in declined:
                    if how == "run-raises":
                        raise RuntimeError(f"injected kernel failure {self.key}")
                    return False
                return real_run(self)

            monkeypatch.setattr(ChunkJob, "run", run)
        counts = _composite_counts(pooled)
        debug_log.clear()

        _assert_same(pooled.render_rect(*rect, mip=0), serial.render_rect(*rect, mip=0), how)
        assert list(pooled._cache) == list(serial._cache), f"{how}: LRU order differs"
        assert counts == dict.fromkeys(declined, 1), f"{how}: synchronous composites {counts}"
        if how == "run-raises":
            log = debug_log.get_log_text()
            where = {"pool": "", "worker": " on a worker", "inline": " inline"}[path]
            assert all(f"chunk {key} raised{where}" in log for key in declined), log


# --- T5: gating ------------------------------------------------------------------


def test_pooling_engages_only_where_it_can(native_kernel, sprite_install, fake_farm, synthetic_textures, pool):  # noqa: F811
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        cache = _make("stepped", scenario)
        executor = _CountingExecutor(pool)
        cache.set_chunk_executor(executor, WORKERS)
        keys, rect = _block(cache, 0, 3, 2)
        for key in keys[1:]:
            cache.get_chunk(*key)
        cache.render_rect(*rect, mip=0)
        assert executor.jobs == [], "one missing chunk was pooled"

        cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
        cache.render_rect(*rect, mip=0)
        assert len(executor.jobs) == len(keys), "control: two or more missing chunks must pool"

        # Flat and numpy prepare nothing a worker could run, so gating shows as no prepare at all.
        executor.jobs.clear()
        prepared = []
        real_prepare = cache.prepare_chunk_job
        cache.prepare_chunk_job = lambda *a, **k: prepared.append(a) or real_prepare(*a, **k)
        cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
        with composite_backend.use_backend("numpy"):
            cache.render_rect(*rect, mip=0)
        assert prepared == [] and executor.jobs == [], "the numpy backend pooled"

        cache.set_chunk_executor(None, 0)
        cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
        cache.render_rect(*rect, mip=0)
        assert prepared == [] and executor.jobs == [], "no executor still prepared jobs"

        mm = scenario.map_manager
        flat = FlatChunkCache(scenario, render.tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True)
        flat.set_chunk_executor(executor, WORKERS)
        flat_prepared = []
        real_flat_prepare = flat.prepare_chunk_job
        flat.prepare_chunk_job = lambda *a, **k: flat_prepared.append(a) or real_flat_prepare(*a, **k)
        _canvas(flat, 0)
        assert flat_prepared == [] and executor.jobs == [], "Flat pooled"

        cache.set_chunk_executor(executor, 0)
        assert cache._chunk_executor is None, "workers=0 must leave the cache serial"


# --- T8: the join -----------------------------------------------------------------


class _NeverStarts:
    """A pool that never runs anything: every job must come back inline."""

    class _Future(Future):
        def result(self, timeout=None):
            assert self.done(), "the join waited on a job that will never start"
            return super().result(timeout)

    def __init__(self):
        self.futures = []
        self.inputs = []

    def submit(self, fn):
        future = self._Future()
        self.futures.append(future)
        self.inputs.append((fn.__self__, _arrays(fn.__self__.args)))
        return future


def _arrays(value) -> list[np.ndarray]:
    """Copies of every ndarray reachable through nested tuples."""
    if isinstance(value, np.ndarray):
        return [value.copy()]
    if isinstance(value, tuple):
        return [a for item in value for a in _arrays(item)]
    return []


class _Inline:
    """Runs each job at submit and hands back a completed future."""

    def __init__(self):
        self.keys = []

    def submit(self, fn):
        self.keys.append(fn.__self__.key)
        future = Future()
        future.set_result(fn())
        return future


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_pool_that_never_starts_a_job_runs_them_all_inline(
    native_kernel, sprite_install, fake_farm, synthetic_textures, kernel_threads, monkeypatch, style,  # noqa: F811
):
    """Every prepare runs before any kernel here, on a cold pack, so an
    in-place write to an earlier job's inputs by a later prepare shows: each
    job's arrays must still equal their copies taken at its submit."""
    scenario = _scenario_for(style)
    with composite_backend.use_backend("native"):
        cache, serial = _make(style, scenario), _make(style, scenario)
        executor = _NeverStarts()
        cache.set_chunk_executor(executor, WORKERS)
        checked, written = [], []
        real_run = ChunkJob.run

        # Recorded, not asserted: the join would log a raise as a decline.
        def run(self):
            for job, copies in executor.inputs:
                if job is self:
                    if not all(np.array_equal(a, b) for a, b in zip(_arrays(self.args), copies, strict=True)):
                        written.append(self.key)
                    checked.append(self.key)
            return real_run(self)

        monkeypatch.setattr(ChunkJob, "run", run)
        for mip in MIPS:
            _assert_same(_canvas(cache, mip), _canvas(serial, mip), f"{style} mip {mip}, never-starting pool")
        assert executor.futures and all(f.cancelled() for f in executor.futures)
        assert written == [], f"{len(written)} jobs had an input written in place after their submit"
        assert len(checked) == len(executor.futures)
        assert not _on_a_worker(kernel_threads)


def test_an_inline_executor_with_completed_futures_works(
    native_kernel, sprite_install, fake_farm, synthetic_textures,  # noqa: F811
):
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        cache, serial = _make("stepped", scenario), _make("stepped", scenario)
        executor = _Inline()
        cache.set_chunk_executor(executor, WORKERS)
        landed = _landings(cache)
        for mip in MIPS:
            _assert_same(_canvas(cache, mip), _canvas(serial, mip), f"inline executor mip {mip}")
        assert list(cache._cache) == list(serial._cache)
        assert executor.keys and landed == executor.keys, "a completed future's result was not installed"


def test_the_calling_thread_does_not_take_the_whole_batch_from_a_busy_pool(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, monkeypatch,  # noqa: F811
):
    """Every worker is held by a blocker when the batch is submitted. The
    first inline kernel releases them and then waits for a worker to start a
    paint job (a handshake, no timing), so the join must leave the head of
    the batch to the workers rather than cancel it all up front. Mip -1's
    canvas is one batch, so no later batch can reach the freed workers."""
    scenario = _cliff_scenario()
    main = threading.get_ident()
    started = threading.Barrier(WORKERS + 1)
    release, worker_took_one = threading.Event(), threading.Event()
    ran: list[tuple[tuple, int]] = []
    timed_out = []
    real_run = ChunkJob.run

    def run(self):
        ident = threading.get_ident()
        ran.append((self.key, ident))
        if ident == main and not release.is_set():
            release.set()
            # Recorded, not raised: the join would log a raise as a decline.
            if not worker_took_one.wait(10):
                timed_out.append(self.key)
        elif threading.current_thread().name.startswith("test-pooled-paint"):
            worker_took_one.set()
        return real_run(self)

    monkeypatch.setattr(ChunkJob, "run", run)

    def blocker():
        started.wait(10)
        release.wait(10)

    with composite_backend.use_backend("native"):
        cache, serial = _make("stepped", scenario), _make("stepped", scenario)
        cache.set_chunk_executor(pool, WORKERS)
        cache._level(-1)
        for _ in range(WORKERS):
            pool.submit(blocker)
        started.wait(10)
        _assert_same(_canvas(cache, -1), _canvas(serial, -1), "busy pool")
    assert release.is_set(), "no job ran inline -- the pool wasn't busy"
    assert len({key for key, _ident in ran}) <= cache.POOL_BATCH, "more than one batch -- the handshake is moot"
    assert not timed_out, "no worker picked up a paint job once released"
    assert any(ident != main for _key, ident in ran), "every paint job ran on the calling thread"
    assert list(cache._cache) == list(serial._cache)


# --- T12: the memory bound ------------------------------------------------------------


def test_a_batch_shares_one_elevations_snapshot_and_never_copies_the_pack(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, monkeypatch,  # noqa: F811
):
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        cache = _make("stepped", scenario)
        executor = _CountingExecutor(pool)
        cache.set_chunk_executor(executor, WORKERS)
        # Mip -1's whole canvas is one batch: a second batch's own pre-derive may rightly copy.
        cache._level(-1)
        pack = cache._native_units(cache._levels[-1], -1)
        assert pack.pending().size, "the pack is not cold -- vacuous"
        copies = []
        real_unshare = native_composite.UnitPack._unshare

        def unshare(self):
            if self._shared:
                copies.append(self)
            real_unshare(self)

        monkeypatch.setattr(native_composite.UnitPack, "_unshare", unshare)
        derived_by: dict[int, int] = {}
        real_write = native_composite.UnitPack._write_tiles
        calls = []

        def write_tiles(self, tiles):
            tiles = list(tiles)
            calls.append(len(executor.jobs))
            derived_by[len(executor.jobs)] = derived_by.get(len(executor.jobs), 0) + len(tiles)
            real_write(self, tiles)

        monkeypatch.setattr(native_composite.UnitPack, "_write_tiles", write_tiles)

        _canvas(cache, -1)
        jobs = executor.jobs
        assert 2 <= len(jobs) <= cache.POOL_BATCH
        elevs = {id(job.args[6]) for job in jobs}
        assert len(elevs) == 1, f"a batch's Stepped jobs hold {len(elevs)} elevations arrays, not one"
        assert jobs[0].args[6] is not cache.elevations, "the batch shares the live array, not a snapshot"
        assert np.array_equal(jobs[0].args[6], cache.elevations)
        indexes = {id(job.args[-3][0]) for job in jobs}
        assert copies == [], f"{len(copies)} copy-on-write copies of the pack inside one batch"
        assert indexes == {id(pack.index)}, "a job holds a replaced pack.index"
        assert set(calls) <= {0}, f"a prepare derived pack tiles after the first submit ({derived_by})"

        key = _chunk_of_tile(cache, EDIT_X, EDIT_Y)
        cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
        warm_job = cache.prepare_chunk_job(*key)
        assert warm_job.args[6] is not cache.elevations and warm_job.args[6] is not jobs[0].args[6], (
            "a margin-warm job must own its own elevations copy"
        )


# --- T9: a margin-warm job in flight --------------------------------------------------


def test_a_margin_warm_result_for_a_chunk_the_paint_built_is_refused(
    native_kernel, sprite_install, fake_farm, synthetic_textures,  # noqa: F811
):
    import time

    from PyQt5.QtWidgets import QApplication

    from descape import margin_warm

    import conftest

    conftest.ensure_qapp()
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        cache, serial = _make("stepped", scenario), _make("stepped", scenario)
        cache.set_chunk_executor(*margin_warm.chunk_pool())
        keys, rect = _block(cache, 0, 3, 2)
        cache._level(0)
        refused, landed = [], []
        real_install = cache.install_chunk

        def install(job, painted):
            ok = real_install(job, painted)
            (landed if ok else refused).append(job.key)
            return ok

        cache.install_chunk = install
        warmer = margin_warm.MarginWarmer()
        warmer.start(cache, 0, [key[1:] for key in keys[:2]])
        warmer.tick()
        assert warmer._in_flight, "no margin-warm job in flight -- vacuous"
        in_flight = {(0, *xy) for xy in warmer._in_flight}

        _assert_same(cache.render_rect(*rect, mip=0), serial.render_rect(*rect, mip=0), "paint over a warm")
        landed.clear()
        deadline = time.monotonic() + 30
        while warmer.is_active:
            assert time.monotonic() < deadline, "the margin warm never drained"
            QApplication.processEvents()
            time.sleep(0.001)
        assert in_flight <= set(refused), f"a late margin-warm result was not refused ({refused})"
        assert landed == []
        for key in keys:
            assert np.array_equal(cache._cache[key], serial.get_chunk(*key)), key


# --- T11: Perf Trace -------------------------------------------------------------------


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_pooled_repaint_traces_prepare_and_join_and_a_serial_one_composite(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool, style,  # noqa: F811
):
    scenario = _scenario_for(style)
    perf_trace.enable(True)
    perf_trace.reset()
    try:
        with composite_backend.use_backend("native"):
            for executor in (pool, None):
                cache = _make(style, scenario)
                cache.set_chunk_executor(executor, WORKERS if executor else 0)
                cache._level(0)
                perf_trace.reset()
                with perf_trace.phase("repaint"):
                    _canvas(cache, 0)
                phases = dict(perf_trace._repaint_phases)
                if executor is not None:
                    assert {"prepare", "join"} <= phases.keys(), phases
                    assert phases["join"][0] >= 1 and phases["prepare"][0] >= 2, phases
                else:
                    assert "composite" in phases and not {"prepare", "join"} & phases.keys(), phases
    finally:
        perf_trace.enable(False)
        perf_trace.reset()


def test_a_batchs_unit_pack_derive_is_traced_as_prepare(
    native_kernel, sprite_install, fake_farm, synthetic_textures, pool,  # noqa: F811
):
    """A serial composite derives pack tiles inside `composite`; a pooled
    batch derives them up front, which must not fall into `other`."""
    scenario = _cliff_scenario()
    perf_trace.enable(True)
    try:
        with composite_backend.use_backend("native"):
            cache = _make("stepped", scenario)
            executor = _CountingExecutor(pool)
            cache.set_chunk_executor(executor, WORKERS)
            cache._level(-1)
            prepared = []
            real_prepare = cache._prepare_rect
            cache._prepare_rect = lambda *a, **k: prepared.append(a) or real_prepare(*a, **k)
            for cold in (True, False):
                perf_trace.reset()
                prepared.clear()
                cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
                assert bool(cache._native_units(cache._levels[-1], -1).pending().size) is cold
                with perf_trace.phase("repaint"):
                    _canvas(cache, -1)
                calls = perf_trace._repaint_phases["prepare"][0]
                assert calls == len(prepared) + cold, f"cold={cold}: {calls} prepare calls for {len(prepared)} chunks"
    finally:
        perf_trace.enable(False)
        perf_trace.reset()


# --- T10: the viewer ----------------------------------------------------------------------

VIEWER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _executor_at_paint(monkeypatch) -> list:
    """(cache style, its chunk executor) at the start of every real paint()."""
    from descape.viewer_canvas import MapCanvasItem

    seen: list = []
    original = MapCanvasItem.paint

    def spy(self, painter, option, widget=None):
        seen.append((self._cache.style, self._cache._chunk_executor))
        return original(self, painter, option, widget)

    monkeypatch.setattr(MapCanvasItem, "paint", spy)
    return seen


@pytest.mark.gui
def test_the_viewer_hands_its_cache_the_shared_pool_before_the_first_paint(native_kernel, monkeypatch):  # noqa: F811
    from PyQt5.QtWidgets import QApplication

    from descape import margin_warm
    from descape.map_view import MapView

    import conftest

    seen = _executor_at_paint(monkeypatch)
    # Offscreen no paint fires inside set_source(), so also check the executor at its entry.
    at_set_source = []
    real_set_source = MapView.set_source

    def set_source_spy(self, *args, **kwargs):
        cache = kwargs.get("cache", args[2] if len(args) > 2 else None)
        at_set_source.append(getattr(cache, "_chunk_executor", None))
        return real_set_source(self, *args, **kwargs)

    monkeypatch.setattr(MapView, "set_source", set_source_spy)
    window = conftest.stepped_window(VIEWER_FIXTURE, size=300)
    try:
        pool = margin_warm._pool()
        assert at_set_source and all(e is pool for e in at_set_source), f"set_source() before the pool: {at_set_source}"
        assert window._cache.style == "stepped" and window._cache._chunk_executor is pool
        assert seen, "no paint ran -- vacuous"
        assert all(executor is pool for _style, executor in seen), f"a paint ran before the pool was set: {seen[:3]}"
        for style in ("Sloped", "Flat", "Stepped"):
            seen.clear()
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            # Flat with Isometric View on renders through the Stepped cache.
            assert window._cache.style == window._render_style and window._cache._chunk_executor is pool, style
            assert seen and all(executor is pool for _style, executor in seen), f"{style}: {seen[:3]}"
            assert all(e is pool for e in at_set_source), f"{style}: set_source() before the pool"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_no_workers_leaves_the_viewer_serial(native_kernel, monkeypatch):  # noqa: F811
    from descape import margin_warm

    import conftest

    monkeypatch.setattr(margin_warm, "WORKERS", 0)
    assert margin_warm.chunk_pool() == (None, 0)
    window = conftest.stepped_window(VIEWER_FIXTURE, size=300)
    try:
        assert window._cache._chunk_executor is None
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_pooled_scene_render_at_a_coarser_mip_equals_a_serial_one(native_kernel):  # noqa: F811
    from PyQt5.QtCore import QRectF
    from test_mip_viewer import _render_scene_rect

    from descape import composite_backend, margin_warm

    import conftest

    if composite_backend.native is None:
        pytest.skip("numpy backend forced: render_rect() never pools there (T5 pins that gate)")
    window = conftest.stepped_window(VIEWER_FIXTURE, size=300)
    try:
        cache, item = window._cache, window.map_view._canvas_item
        if cache.mip_levels()[0] >= 0:
            pytest.skip("fixture enumerates no level coarser than the reference")
        window._cancel_warms()
        landed = _landings(cache)
        scene, rect = window.map_view.scene(), QRectF(0, 0, *cache.canvas_dims(0))
        renders = {}
        for name, pool in (("pooled", margin_warm.chunk_pool()), ("serial", (None, 0))):
            cache.set_chunk_executor(*pool)
            cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
            landed.clear()
            renders[name] = _render_scene_rect(scene, rect, 0.5)
            assert item._last_mip < 0, f"{name}: the render selected mip {item._last_mip}, not a coarser one"
            assert bool(landed) is (name == "pooled"), f"{name}: {len(landed)} pooled chunks landed"
        _assert_same(renders["pooled"], renders["serial"], "scene render at a coarser mip")
    finally:
        conftest.close_window(window)
