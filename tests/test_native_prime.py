"""Native input priming (TASK-031.85): native_composite.prime_walk(), the
caches' native_prime_job() and its LevelWarmer chaining.

Priming only runs cached builders earlier than a first prepare would, so
the contract is: after it, a first chunk builds nothing it primed, and its
bytes equal an unprimed chunk's. Cold builds are counted as lru misses (one
miss is one call of the `__wrapped__` builder); every test that counts them
clears those caches first, since they are process-wide.

The maps are test_terrain_blend_render's synthetic blend install: a 120x120
map, ladder 16-192 px at base 64, 351 blend keys and 70 mask pairs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_farm_terrain import fake_farm  # noqa: F401 -- a fixture
from test_native_composite import native_kernel, synthetic_textures  # noqa: F401 -- fixtures
from test_sprite_edit_bbox import sprite_install  # noqa: F401 -- a fixture
from test_terrain_blend_render import _cache as _blend_cache
from test_terrain_blend_render import _scenario as _blend_scenario
from test_terrain_blend_render import blend_install  # noqa: F401 -- a fixture

from descape import (
    asset_source,
    composite_backend,
    debug_log,
    iso_geometry,
    level_warm,
    native_composite,
    render_cache_leveled,
    render_iso,
    render_paint,
    render_sloped,
    terrain_blend,
)
from descape.render_cache import FlatChunkCache, IsoChunkCache
from descape.view_layers import LayerState
from testkit.render_family import RENDER, current, set_render_cache_global, set_render_global

KINDS = ("stepped", "sloped", "iso-flat")
# The seed paint at 16 px derives every blend key (and loads every layer mask); 32 px is primed.
SEED_MIP, TARGET_MIP = -2, -1
PRIMED = (
    *((iso_geometry, name) for name in (
        "diamond_indices", "seam_edge_indices", "seam_apex_indices", "tile_edge_indices", "skirt_quad_indices",
        "shadow_quad_indices", "shadow_apex_indices", "_sloped_quad_nesw", "_sloped_quad_nwse",
        "sloped_tile_edge_indices",
    )),
    (native_composite, "stepped_tables"),
    (native_composite, "sloped_shapes"),
    *((RENDER, name) for name in ("_seam_factors", "_shadow_factors", "_skirt_lut", "_slope_shade")),
    (asset_source, "get_blend_mask"),
)


def _cold() -> None:
    for owner, name in PRIMED:
        current(owner, name).cache_clear()


def _misses() -> dict[str, int]:
    return {name: current(owner, name).cache_info().misses for owner, name in PRIMED}


def _fills(before: dict[str, int]) -> dict[str, int]:
    return {k: v - before[k] for k, v in _misses().items() if v != before[k]}


def _make(kind: str, scenario=None):
    if kind != "iso-flat":
        return _blend_cache(kind, scenario if scenario is not None else _blend_scenario(kind))
    # Flat + Isometric View: an IsoChunkCache over zero elevations, as the viewer builds it.
    scenario = scenario if scenario is not None else _blend_scenario("stepped")
    elevations, proj = render_iso.elevations_and_proj(scenario)
    mm = scenario.map_manager
    tile_px = render_paint.tile_pixels_for_map(mm.map_width, mm.map_height)
    return IsoChunkCache(scenario, np.zeros_like(elevations), proj, tile_px, sprites=False, layers=LayerState())


def _seeded(kind: str, scenario=None):
    cache = _make(kind, scenario)
    cache.render_rect(0, 0, *cache.canvas_dims(SEED_MIP), mip=SEED_MIP)
    assert cache._blend_key_map.keys, "vacuous: the seed paint derived no blend key"
    assert not cache.is_level_resident(TARGET_MIP), "vacuous: the target level is built already"
    return cache


def _warm(cache, mips) -> level_warm.LevelWarmer:
    warmer = level_warm.LevelWarmer()
    warmer.start(cache, mips)
    warmer.run_to_completion()
    return warmer


def _level(cache, mip: int = TARGET_MIP) -> np.ndarray:
    return cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)


def _shape_count(cache) -> int:
    return len(native_composite.sloped_shapes(cache.mip_tile_px(TARGET_MIP)).corners)


def _arm(kind: str, scenario) -> tuple[np.ndarray, dict[str, int], int]:
    """(target level's pixels, cold fills while compositing it, new Sloped
    shapes registered by it) after a level warm of the target, from cold caches."""
    _cold()
    cache = _seeded(kind, scenario)
    _warm(cache, [TARGET_MIP])
    assert cache.is_level_resident(TARGET_MIP)
    before = _misses()
    sloped = kind == "sloped"
    # After the snapshot: reading the count creates the registry on an unprimed arm.
    shapes = _shape_count(cache) if sloped else 0
    image = _level(cache)
    return image, _fills(before), (_shape_count(cache) if sloped else 0) - shapes


def _errors(monkeypatch) -> list[str]:
    """Every ERROR debug_log line from here on (LevelWarmer logs a dropped job there)."""
    seen: list[str] = []
    real = debug_log.log

    def log(message, level=debug_log.INFO):
        if level == debug_log.ERROR:
            seen.append(message)
        real(message, level)

    monkeypatch.setattr(debug_log, "log", log)
    return seen


# --- zero cold builds and byte identity ---------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_primed_level_builds_nothing_primed_while_compositing(blend_install, native_kernel, monkeypatch, kind) -> None:  # noqa: F811
    """After a level warm drains, compositing every chunk of that level
    builds none of the factory, producers, factors, shades or primed masks,
    and (Sloped) registers no shape. The unprimed control builds them."""
    with composite_backend.use_backend("native"):
        scenario = _make(kind).scenario
        _image, primed, new_shapes = _arm(kind, scenario)
        set_render_cache_global(monkeypatch, "WARM_NATIVE_INPUTS", False)
        _image, control, _ = _arm(kind, scenario)
    first = "sloped_shapes" if kind == "sloped" else "stepped_tables"
    assert first in control and "get_blend_mask" in control, f"vacuous: the control built {control}"
    assert primed == {}, f"cold builds after priming: {primed}"
    assert new_shapes == 0


@pytest.mark.parametrize("kind", KINDS)
def test_priming_changes_no_pixel(blend_install, native_kernel, monkeypatch, kind) -> None:  # noqa: F811
    """The whole target level composited after priming equals the same level
    composited with WARM_NATIVE_INPUTS off."""
    with composite_backend.use_backend("native"):
        scenario = _make(kind).scenario
        primed, _fills_on, _ = _arm(kind, scenario)
        set_render_cache_global(monkeypatch, "WARM_NATIVE_INPUTS", False)
        control, _fills_off, _ = _arm(kind, scenario)
    assert primed.shape == control.shape
    assert np.array_equal(primed, control)


def test_sloped_priming_interleaved_with_prepares_changes_no_pixel(blend_install, native_kernel, monkeypatch) -> None:  # noqa: F811
    """Chunks composited between the walk's slices register shapes in their
    own order, so ids differ from a clean walk's; the pixels may not."""
    rebuilds = []
    real_packed = native_composite.SlopedShapes.packed

    def packed(self):
        if self._packed is None:
            rebuilds.append(len(self.corners))
        return real_packed(self)

    monkeypatch.setattr(native_composite.SlopedShapes, "packed", packed)
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)  # one step a tick
    with composite_backend.use_backend("native"):
        _cold()
        cache = _seeded("sloped")
        cols, rows = (-(-d // cache.chunk_px) for d in cache.canvas_dims(TARGET_MIP))
        chunks = [(cx, cy) for cy in range(rows) for cx in range(cols)]
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [TARGET_MIP])
        interleaved = 0
        while warmer.tick():
            if warmer._job is not None and warmer._job.kind == "native" and interleaved < len(chunks):
                cache.get_chunk(TARGET_MIP, *chunks[interleaved])
                interleaved += 1
        interleaved_image = _level(cache)
        set_render_cache_global(monkeypatch, "WARM_NATIVE_INPUTS", False)
        control, _fills, _ = _arm("sloped", cache.scenario)
    assert interleaved >= 4, f"vacuous: {interleaved} chunks between slices"
    assert len(rebuilds) >= 2, f"vacuous: the walk and the chunks packed {rebuilds}"
    assert np.array_equal(interleaved_image, control)


# --- the job and its chaining ---------------------------------------------------


def test_start_notifies_each_mip_once_after_its_native_job(blend_install, native_kernel, monkeypatch) -> None:  # noqa: F811
    """The notify moves to the native job, each mip's last: on_job_done fires
    once per mip, with that mip's walk already finished."""
    with composite_backend.use_backend("native"):
        cache = _make("stepped")
        cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)
        finished: set[int] = set()
        real = cache.native_prime_job

        def spy(mip, after_level_warm=False):
            job = real(mip, after_level_warm)
            if job is None:
                return None

            def gen():
                yield from job.gen
                finished.add(mip)

            return replace(job, gen=gen())

        monkeypatch.setattr(cache, "native_prime_job", spy)
        warmer = level_warm.LevelWarmer()
        calls = []
        notifying = warmer.start(cache, [-2, -1], on_job_done=lambda m: calls.append((m, m in finished)))
        kinds = [(m, job.kind) for m, job, _notify in warmer._queue]
        warmer.run_to_completion()
    assert (-2, "native") in kinds and (-1, "native") in kinds, f"vacuous: {kinds}"
    assert notifying == {-2, -1}
    assert calls == [(-2, True), (-1, True)]


def test_no_native_job_on_numpy_top_down_flat_or_the_switch_off(blend_install, native_kernel, monkeypatch) -> None:  # noqa: F811
    with composite_backend.use_backend("native"):
        cache = _make("stepped")
        assert cache.native_prime_job(TARGET_MIP, after_level_warm=True) is not None, "control: native, switch on"
        flat = FlatChunkCache(cache.scenario, cache.tile_px, sprites=True)
        assert all(flat.native_prime_job(m, after_level_warm=True) is None for m in flat.mip_levels())
    with composite_backend.use_backend("numpy"):
        assert cache.native_prime_job(TARGET_MIP, after_level_warm=True) is None
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [TARGET_MIP])
        assert not [job for _m, job, _n in warmer._queue if job.kind == "native"]
        warmer.cancel()
    set_render_cache_global(monkeypatch, "WARM_NATIVE_INPUTS", False)
    with composite_backend.use_backend("native"):
        assert cache.native_prime_job(TARGET_MIP, after_level_warm=True) is None
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [TARGET_MIP])
        assert not [job for _m, job, _n in warmer._queue if job.kind == "native"]
        warmer.cancel()


def test_levels_above_the_cap_get_no_native_job(blend_install, native_kernel) -> None:  # noqa: F811
    with composite_backend.use_backend("native"):
        cache = _make("sloped")
        by_px = {cache.mip_tile_px(m): m for m in cache.mip_levels()}
        cap = render_cache_leveled.NATIVE_PRIME_MAX_TILE_PX
        assert any(px > cap for px in by_px) and any(px <= cap for px in by_px), f"vacuous ladder {sorted(by_px)}"
        for px, mip in by_px.items():
            job = cache.native_prime_job(mip, after_level_warm=True)
            assert (job is None) == (px > cap), f"{px} px"
        above = min(px for px in by_px if px > cap)
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [by_px[above], by_px[cap]])
        queued = {(cache.mip_tile_px(m), job.kind) for m, job, _n in warmer._queue}
        warmer.cancel()
    assert (cap, "native") in queued and (above, "walk") in queued
    assert (above, "native") not in queued


def test_a_pending_level_chains_flush_pack_then_native(native_kernel, sprite_install) -> None:  # noqa: F811
    """A flush job is a level job, so an elevation stroke's deferred levels get
    flush, pack, native, the notify on native (the 2026-11-09 option A rule)."""
    from test_stepped_defer_offscreen import OFFSCREEN, _resident_cache, _stroke
    from test_stepped_defer_offscreen import _scenario as _defer_scenario

    with composite_backend.use_backend("native"):
        scenario, _units = _defer_scenario()
        cache = _resident_cache(scenario)
        _stroke(cache, scenario)
        mips = [m for m in OFFSCREEN if cache.mip_tile_px(m) <= render_cache_leveled.NATIVE_PRIME_MAX_TILE_PX]
        assert mips and all(cache._levels[m].pending_elev for m in mips), "vacuous: no deferred level"
        warmer = level_warm.LevelWarmer()
        notifying = warmer.start(cache, mips)
        queued = [(m, job.kind, notify) for m, job, notify in warmer._queue]
        warmer.cancel()
    assert queued == [entry for m in mips for entry in ((m, "flush", False), (m, "pack", False), (m, "native", True))]
    assert notifying == set(mips)


def test_a_re_arm_over_a_resident_primed_level_queues_no_native_job(blend_install, native_kernel) -> None:  # noqa: F811
    """A completed walk marks its level (TASK-031.82 D6), so the re-arm after
    it queues nothing and starts no timer."""
    with composite_backend.use_backend("native"):
        cache = _seeded("stepped")
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [TARGET_MIP])
        assert [job.kind for _m, job, _n in warmer._queue][-1] == "native", "vacuous: the first warm queued none"
        warmer.run_to_completion()
        assert cache.is_level_resident(TARGET_MIP) and cache.native_primed(TARGET_MIP)
        assert cache.native_prime_job(TARGET_MIP) is None
        assert warmer.start(cache, [TARGET_MIP]) == set() and not warmer.is_active
        assert not [job for _m, job, _n in warmer._queue if job.kind == "native"]
        warmer.cancel()


# --- re-queue on re-arm (TASK-031.82 D6) ----------------------------------------


def test_level_warmed_is_a_composited_chunk_or_a_finished_prime(blend_install, native_kernel, monkeypatch) -> None:  # noqa: F811
    """margin_warm's cold-first defer reads it (TASK-031.82 D3)."""
    with composite_backend.use_backend("native"):
        cache = _seeded("stepped")
        assert cache.level_warmed(SEED_MIP), "the seed paint composited chunks there"
        assert not cache.level_warmed(TARGET_MIP)
        _warm(cache, [TARGET_MIP])
        assert cache.level_warmed(TARGET_MIP) and cache.native_primed(TARGET_MIP)
        assert not cache.level_warmed(0)
        cache.get_chunk(0, 0, 0)
        assert cache.level_warmed(0) and not cache.native_primed(0)
        flat = FlatChunkCache(cache.scenario, cache.tile_px, sprites=False)
        mip = flat.mip_levels()[0]
        assert not flat.level_warmed(mip)
        flat.get_chunk(mip, 0, 0)
        assert flat.level_warmed(mip)


def test_a_walk_cancelled_midway_is_re_queued_by_the_next_re_arm(blend_install, native_kernel, monkeypatch) -> None:  # noqa: F811
    """A cold roll's crossing cancels the queued prime; the re-arm over the
    now-resident level queues it again, alone, notifying once after it, and a
    finished walk marks the level so the next re-arm is quiet."""
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)
    with composite_backend.use_backend("native"):
        cache = _seeded("stepped")
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [TARGET_MIP])
        _tick_into_native(warmer, 2)
        warmer.cancel()
        assert cache.is_level_resident(TARGET_MIP) and not cache.native_primed(TARGET_MIP)
        calls = []
        notifying = warmer.start(cache, [TARGET_MIP], on_job_done=calls.append)
        assert [(m, job.kind, n) for m, job, n in warmer._queue] == [(TARGET_MIP, "native", True)]
        assert notifying == {TARGET_MIP}
        assert warmer.is_active, "the accepted cost: a resident, unmarked level's re-arm starts a warm"
        warmer.run_to_completion()
        assert calls == [TARGET_MIP]
        assert cache.native_primed(TARGET_MIP)
        assert warmer.start(cache, [TARGET_MIP]) == set() and not warmer.is_active


def test_requeue_prime_off_leaves_a_resident_level_with_no_job(blend_install, native_kernel) -> None:  # noqa: F811
    """The viewer's pending level: resident and unprimed, its load warm must not
    wait behind a re-queued prime, so start() queues none (a level job still chains one)."""
    with composite_backend.use_backend("native"):
        cache = _seeded("stepped")
        cache._level(TARGET_MIP)
        assert cache.native_prime_job(TARGET_MIP) is not None, "vacuous: nothing to re-queue"
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [TARGET_MIP], requeue_prime=False)
        assert "native" not in [job.kind for _m, job, _n in warmer._queue]
        cache._refresh_all_units()
        warmer.start(cache, [TARGET_MIP], requeue_prime=False)
        kinds = [job.kind for _m, job, _n in warmer._queue]
        warmer.cancel()
    assert kinds[-1] == "native", f"a level job still chains its prime: {kinds}"


def test_a_source_gen_bump_re_queues_and_a_rebuild_mid_walk_records_nothing(blend_install, native_kernel, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)
    with composite_backend.use_backend("native"):
        cache = _seeded("stepped")
        _warm(cache, [TARGET_MIP])
        assert cache.native_primed(TARGET_MIP)
        # A bump the paint rebuilds under: resident again, the marker one gen behind.
        cache._refresh_all_units()
        cache._level(TARGET_MIP)  # a paint's rebuild
        assert cache.is_level_resident(TARGET_MIP) and not cache.native_primed(TARGET_MIP)
        assert cache.native_prime_job(TARGET_MIP) is not None

        # The level rebuilt between atoms while residency holds: the walk ends, marks nothing.
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [TARGET_MIP])
        _tick_into_native(warmer, 2)
        cache._refresh_all_units()
        cache._level(TARGET_MIP)
        assert cache.is_level_resident(TARGET_MIP), "vacuous: the rebuild left the level stale"
        warmer.run_to_completion()
        assert not cache.native_primed(TARGET_MIP)
        assert cache.native_prime_job(TARGET_MIP) is not None


def test_a_pack_only_mip_gets_pack_then_native_both_last_and_silent(native_kernel, sprite_install) -> None:  # noqa: F811
    """The anchor (pack_only) queues its derive then its prime after every
    neighbour's chain, neither notifying."""
    from test_stepped_defer_offscreen import OFFSCREEN, _resident_cache
    from test_stepped_defer_offscreen import _scenario as _defer_scenario

    with composite_backend.use_backend("native"):
        scenario, _units = _defer_scenario()
        cache = _resident_cache(scenario)
        neighbour, anchor = OFFSCREEN
        assert all(cache.mip_tile_px(m) <= render_cache_leveled.NATIVE_PRIME_MAX_TILE_PX for m in OFFSCREEN)
        cache._levels[anchor].unit_pack = None  # as a wholesale rebuild leaves it: the derive has work
        assert cache.pack_warm_job(anchor) is not None, "vacuous: the anchor has no derive to queue"
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [neighbour], pack_only=(anchor,))
        queued = [(m, job.kind, n) for m, job, n in warmer._queue]
        warmer.cancel()
    assert queued == [(neighbour, "native", True), (anchor, "pack", False), (anchor, "native", False)]


def test_the_switch_off_queues_no_re_arm_prime(blend_install, native_kernel, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)
    with composite_backend.use_backend("native"):
        cache = _seeded("stepped")
        _level(cache)  # the paint builds the level: resident, never primed
        assert cache.is_level_resident(TARGET_MIP) and not cache.native_primed(TARGET_MIP)
        assert cache.native_prime_job(TARGET_MIP) is not None, "control: the switch on re-queues"
        set_render_cache_global(monkeypatch, "WARM_NATIVE_INPUTS", False)
        assert cache.native_prime_job(TARGET_MIP) is None
        warmer = level_warm.LevelWarmer()
        assert warmer.start(cache, [TARGET_MIP], pack_only=(SEED_MIP,)) == set()
        assert not [job for _m, job, _n in warmer._queue if job.kind == "native"]
        warmer.cancel()


# --- cancel and staleness ----------------------------------------------------


def _tick_into_native(warmer, steps: int) -> None:
    """Ticks (one step each, BUDGET_MS 0) until the native job has run `steps` steps."""
    ran = 0
    while warmer.tick():
        if warmer._job is not None and warmer._job.kind == "native":
            ran += 1
            if ran >= steps:
                return
    pytest.fail("the native job never ran")


@pytest.mark.parametrize("kind", ("stepped", "sloped"))
def test_a_cancelled_or_stale_walk_stops_harmlessly(blend_install, native_kernel, monkeypatch, kind) -> None:  # noqa: F811
    """cancel() mid-walk, a KeyMap reset between slices, the level leaving
    residency, and the warm moving to a replacement cache: nothing raises,
    nothing is logged, a stale walk builds no further atom, and the level
    still composites byte-identically."""
    errors = _errors(monkeypatch)
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)
    factory = "sloped_shapes" if kind == "sloped" else "stepped_tables"
    with composite_backend.use_backend("native"):
        scenario = _make(kind).scenario
        set_render_cache_global(monkeypatch, "WARM_NATIVE_INPUTS", False)
        reference, _f, _s = _arm(kind, scenario)
        set_render_cache_global(monkeypatch, "WARM_NATIVE_INPUTS", True)

        # cancel() mid-walk
        _cold()
        cache = _seeded(kind, scenario)
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [TARGET_MIP])
        _tick_into_native(warmer, 3)
        warmer.cancel()
        assert not warmer.is_active
        assert np.array_equal(_level(cache), reference)

        # The KeyMap's list mutated in place, then reset or replaced, between slices: the
        # walk's snapshot still primes exactly its keys' masks. A short scan spreads it over steps.
        _cold()
        cache = _seeded(kind, scenario)
        monkeypatch.setattr(native_composite, "PRIME_KEY_SCAN", 16)
        read: list = []
        real_keys = cache._prime_blend_keys

        def keys_read():
            keys = real_keys()
            read.append(list(keys or ()))
            return keys

        monkeypatch.setattr(cache, "_prime_blend_keys", keys_read)
        warmer.start(cache, [TARGET_MIP])
        mm = scenario.map_manager
        tile_px = cache.mip_tile_px(TARGET_MIP)
        step, filled0 = 0, None
        while warmer.tick():
            if not read:
                continue
            if filled0 is None:
                filled0 = asset_source.get_blend_mask.cache_info().misses
            step += 1
            live = cache._blend_key_map.keys
            live.clear()
            live.append((0, -1, ((1, 0, 0),)))
            if step == 3:
                cache._blend_key_map = terrain_blend.KeyMap(mm.map_width, mm.map_height)
            else:
                cache._blend_key_map.reset()
        expected = set()
        for key in read[0]:
            layer, pairs = native_composite.blend_mask_args(key, tile_px)
            if layer is None or asset_source.terrain_mask_loaded(*layer):
                expected.update(pairs)
        assert len(read[0]) > 3 * 16 and step > 3 and expected, "vacuous: the scan ended before the mutations"
        assert asset_source.get_blend_mask.cache_info().misses - filled0 == len(expected)
        assert np.array_equal(_level(cache), reference)

        # the level leaves residency mid-walk: the walk stops before its next atom
        _cold()
        cache = _seeded(kind, scenario)
        warmer.start(cache, [TARGET_MIP])
        _tick_into_native(warmer, 1)
        built = _misses()[factory]
        cache._refresh_all_units()
        assert not cache.is_level_resident(TARGET_MIP)
        warmer.run_to_completion()
        assert _misses()[factory] == built, "a stale walk kept building"
        assert np.array_equal(_level(cache), reference)

        # the warm moves to a replacement cache mid-walk
        _cold()
        old = _seeded(kind, scenario)
        warmer.start(old, [TARGET_MIP])
        _tick_into_native(warmer, 3)
        new = _seeded(kind, scenario)
        warmer.start(new, [TARGET_MIP])
        warmer.run_to_completion()
        assert new.is_level_resident(TARGET_MIP)
        assert np.array_equal(_level(new), reference)
        assert np.array_equal(_level(old), reference)
    assert errors == []


# --- the layer-not-loaded skip -------------------------------------------------


def test_masks_of_a_key_whose_layer_is_not_loaded_are_skipped(monkeypatch) -> None:
    """The walk never loads a layer mask: a key whose layer terrain_mask_at()
    has not returned at this size gets none of its masks primed; a key whose
    layer is loaded, and one with no layer, get theirs."""
    tile_px = 32
    size = asset_source.terrain_texture_size(tile_px)
    loads = []
    monkeypatch.setattr(asset_source, "get_blendomatic", _fake_blendomatic)
    monkeypatch.setattr(terrain_blend, "layer_mask_name", lambda tid: f"layer{tid}.png")

    def mask_array(name):
        loads.append(name)
        return np.zeros((asset_source.LOADED_TEXTURE_SIZE,) * 2, dtype=np.uint8)

    mask_array.cache_clear = lambda: None
    monkeypatch.setattr(asset_source, "get_terrain_mask_array", mask_array)
    asset_source.get_blend_mask.cache_clear()
    asset_source.terrain_mask_at("layer2.png", size)  # a paint loaded layer 2 only
    loads.clear()
    unloaded, loaded, bare = (0, 1, ((5, 1, 2),)), (0, 2, ((5, 3, 4),)), (0, -1, ((5, 5, 6),))
    _drain(native_composite._prime_masks([unloaded, loaded, bare], tile_px))
    assert loads == [], "the walk loaded a layer mask"
    assert asset_source.get_blend_mask.cache_info().misses == 2
    filled = asset_source.get_blend_mask.cache_info().misses
    asset_source.get_blend_mask(3, 4, tile_px)
    asset_source.get_blend_mask(5, 6, tile_px)
    assert asset_source.get_blend_mask.cache_info().misses == filled, "a kept key's mask was not primed"
    asset_source.get_blend_mask(1, 2, tile_px)
    assert asset_source.get_blend_mask.cache_info().misses == filled + 1, "an unloaded layer's key was primed"


# --- the input tick ---------------------------------------------------------------

# Worst wall ms per call at 96 px in the TASK-031.85 census (three xgui rounds, in-context where measured):
# fixed ops (slice 1 minus a rise), one rise, assembly, factors, and per item: mask, Sloped
# registration and shade (batch worst / batch size); Sloped init, a 32-row discovery band, packed().
# The walk's own bookkeeping, measured 2026-11-09 on old-allies and June (best of 20, sandbox host):
# a key's mask-argument scan (blend_mask_args, terrain_mask_loaded, dedupe) 0.15-0.17 ms per 512
# keys here, 0.27 ms in the TASK-031.85 review, the larger charged; the keys snapshot 0.0031 ms per
# 3,232 keys; Sloped's first step (heights(), set(ids)) 0.0002 ms, charged 0.01 for a real shape count.
ATOM_MS = {
    "fixed": 0.32, "rise": 0.13, "assembly": 2.10, "factors": 0.32, "mask": 1.27 / 8,
    "sloped_init": 0.22, "band": 0.16, "shape": 1.10 / 8, "packed": 0.73, "shade": 0.19 / 8,
    "scan_key": 0.27 / 512, "snapshot_key": 0.0031 / 3232, "shapes_setup": 0.01,
}
TICK_LIMIT_MS = 6.0


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0
        # Inside a charged builder: its own producer calls are part of its cost.
        self.depth = 0
        self.counts: dict[str, int] = {}

    def __call__(self) -> float:
        return self.t

    def charge(self, atom: str, n: int = 1) -> None:
        if not self.depth:
            self.t += n * ATOM_MS[atom] / 1000.0
            self.counts[atom] = self.counts.get(atom, 0) + n


def _charged(clock: _Clock, monkeypatch, cache) -> None:
    """Every primed builder advances the clock by its census cost on a cold call,
    and the walk's bookkeeping by its measured cost."""
    nc = native_composite
    real_keys, real_shapes = cache._prime_blend_keys, nc._prime_shapes

    def keys_snapshot():
        keys = real_keys()
        clock.charge("snapshot_key", len(keys or ()))
        return keys

    def prime_shapes(tile_px, elev_step, heights):
        def charged():
            # Read in _prime_shapes()' first step, beside set(ids).
            clock.charge("shapes_setup")
            return heights()

        yield from real_shapes(tile_px, elev_step, charged)

    monkeypatch.setattr(cache, "_prime_blend_keys", keys_snapshot)
    monkeypatch.setattr(nc, "_prime_shapes", prime_shapes)

    def on_miss(fn, atom):
        def wrapper(*args):
            misses = fn.cache_info().misses
            clock.depth += 1
            try:
                value = fn(*args)
            finally:
                clock.depth -= 1
            if fn.cache_info().misses > misses:
                clock.charge(atom)
            return value
        # conftest's per-test clear_install_caches() runs before monkeypatch restores.
        wrapper.cache_clear, wrapper.cache_info = fn.cache_clear, fn.cache_info
        return wrapper

    def always(fn, atom):
        def wrapper(*args):
            clock.charge(atom)
            return fn(*args)
        return wrapper

    monkeypatch.setattr(nc, "_fixed_ops", always(nc._fixed_ops, "fixed"))
    monkeypatch.setattr(nc, "_rise_ops", always(nc._rise_ops, "rise"))
    monkeypatch.setattr(nc, "_shape_keys", always(nc._shape_keys, "band"))
    monkeypatch.setattr(nc, "blend_mask_args", always(nc.blend_mask_args, "scan_key"))
    monkeypatch.setattr(nc, "stepped_tables", on_miss(nc.stepped_tables, "assembly"))
    monkeypatch.setattr(nc, "sloped_shapes", on_miss(nc.sloped_shapes, "sloped_init"))
    monkeypatch.setattr(asset_source, "get_blend_mask", on_miss(asset_source.get_blend_mask, "mask"))
    set_render_global(monkeypatch, "_slope_shade", on_miss(render_sloped._slope_shade, "shade"))
    real_tables, real_ids, real_packed = nc.SteppedTables.kernel_tables, nc.SlopedShapes.shape_ids, nc.SlopedShapes.packed

    def kernel_tables(self):
        if self._factor_src is None:
            clock.charge("factors")
        return real_tables(self)

    def shape_ids(self, keys):
        clock.charge("shape", sum(key not in self.ids for key in keys))
        return real_ids(self, keys)

    def packed(self):
        if self._packed is None:
            clock.charge("packed")
        return real_packed(self)

    monkeypatch.setattr(nc.SteppedTables, "kernel_tables", kernel_tables)
    monkeypatch.setattr(nc.SlopedShapes, "shape_ids", shape_ids)
    monkeypatch.setattr(nc.SlopedShapes, "packed", packed)


@pytest.mark.parametrize("kind", ("stepped", "sloped"))
def test_a_native_tick_under_the_input_budget_stays_within_one_atom(blend_install, native_kernel, monkeypatch, kind) -> None:  # noqa: F811
    """With every atom charged its census worst at 96 px on a fake clock, no
    tick of the native job under INPUT_BUDGET_MS runs past TICK_LIMIT_MS
    (the budget plus one atom)."""
    clock = _Clock()
    monkeypatch.setattr(level_warm, "_now", clock)
    monkeypatch.setattr(level_warm, "_last_input", None)  # restored at teardown: note_input() stamps fake time
    with composite_backend.use_backend("native"):
        _cold()
        cache = _make(kind)
        cache.render_rect(0, 0, *cache.canvas_dims(SEED_MIP), mip=SEED_MIP)
        mip = next(m for m in cache.mip_levels() if cache.mip_tile_px(m) == 96)
        _charged(clock, monkeypatch, cache)
        warmer = level_warm.LevelWarmer()
        warmer.start(cache, [mip])
        ticks = []
        while True:
            level_warm.note_input()
            native = warmer._job is not None and warmer._job.kind == "native"
            t0 = clock.t
            more = warmer.tick()
            native = native or (warmer._job is not None and warmer._job.kind == "native")
            if native:
                ticks.append((clock.t - t0) * 1000.0)
                assert warmer._tick_budget_ms == level_warm.INPUT_BUDGET_MS
            if not more:
                break
    assert sum(ticks) > 3 * level_warm.INPUT_BUDGET_MS, f"vacuous: the walk charged {sum(ticks):.2f} ms"
    bookkeeping = ("scan_key", "snapshot_key", "shapes_setup") if kind == "sloped" else ("scan_key", "snapshot_key")
    assert all(clock.counts.get(atom) for atom in bookkeeping), f"bookkeeping not charged: {clock.counts}"
    assert max(ticks) <= TICK_LIMIT_MS, f"a tick ran {max(ticks):.2f} ms: {[round(t, 2) for t in ticks]}"

# The blendomatic's (modes, tiles) without an install: 9 blend modes of 31 tiles (openage doc/media/blendomatic.md).
BLENDOMATIC_SHAPE = (9, 31)


def _blend_key_space() -> tuple[int, int]:
    tiles = asset_source.get_blendomatic()
    return BLENDOMATIC_SHAPE if tiles is None else (int(tiles.shape[0]), int(tiles.shape[1]))


def _every_tile_px(monkeypatch) -> set[int]:
    """Every level tile_px any graphics quality reaches on a small or a large
    map, iso (native tiers) and top-down: a session may switch quality, and the lru outlives it."""
    from descape import settings

    sizes: set[int] = set()
    for quality in settings.GRAPHICS_QUALITY_LABELS:
        # The cached value get_graphics_quality() returns; set_graphics_quality() would also save config.
        monkeypatch.setattr(settings, "_graphics_quality", quality)
        for side in (render_paint.LARGE_MAP_TILE_THRESHOLD, 2 * render_paint.LARGE_MAP_TILE_THRESHOLD):
            base = render_paint.tile_pixels_for_map(side, side)
            finest = render_paint.finest_tile_pixels()
            sizes.update(iso_geometry.mip_tile_px_candidates(
                base, finest, native_tier=True, native_tier_max=render_paint.native_tier_max()).values())
            sizes.update(iso_geometry.mip_tile_px_candidates(base, finest).values())
    return sizes


def test_the_blend_mask_lru_covers_every_quality_ladders_key_space(monkeypatch) -> None:
    """D6: every (mode, tile) at every level any quality and map size reaches
    fits, so no ladder, nor a quality switch, evicts a primed mask."""
    modes, tiles = _blend_key_space()
    sizes = _every_tile_px(monkeypatch)
    assert {16, 96, 128, 192} <= sizes, f"not every ladder: {sorted(sizes)}"
    assert modes * tiles * len(sizes) <= asset_source.get_blend_mask.cache_info().maxsize, sorted(sizes)


def _fake_blendomatic():
    rng = np.random.default_rng(31)
    arr = rng.integers(0, 129, size=(*BLENDOMATIC_SHAPE, asset_source.BLENDOMATIC_TILE_SIZE), dtype=np.uint8)
    arr.flags.writeable = False
    return arr


def _every_pair_key(modes: int, tiles: int) -> list:
    """KeyMap-shaped keys, no layer, naming every (mode, tile) once."""
    return [(0, -1, ((1, mode, tile),)) for mode in range(modes) for tile in range(tiles)]


def _drain(gen) -> int:
    steps = 0
    for _ in gen:
        steps += 1
    return steps


def test_masks_primed_across_three_sizes_survive_a_return_to_the_first(monkeypatch) -> None:
    """D6's LRU control: more than the old 512 distinct masks over three
    sizes, then the first size again refills nothing."""
    monkeypatch.setattr(asset_source, "get_blendomatic", _fake_blendomatic)
    asset_source.get_blend_mask.cache_clear()
    keys = _every_pair_key(*BLENDOMATIC_SHAPE)
    sizes = (16, 32, 64)
    assert len(keys) * len(sizes) > 512, "vacuous: fits the old maxsize"
    for tile_px in sizes:
        _drain(native_composite._prime_masks(keys, tile_px))
    filled = asset_source.get_blend_mask.cache_info().misses
    assert filled == len(keys) * len(sizes)
    _drain(native_composite._prime_masks(keys, sizes[0]))
    assert asset_source.get_blend_mask.cache_info().misses == filled, "a return to the first size refilled masks"
