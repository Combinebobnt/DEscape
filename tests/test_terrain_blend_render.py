"""TASK-143 step 3: terrain blending wired through every render path.

A synthetic blendomatic (random own-weight alphas), blend table and layer
mask stand in for the install, so blend_effective() holds without one. Every
cache comparison runs under both composite backends.

Oracles: chunks stitched == a full render, a one-tile terrain or layer edit
patched through the viewer's own funnel == a fresh cache (whose mirrors are
read from the tiles), the blending row off == blend=False == no blendomatic.
Step 4: the kernel composes blended top blocks itself, byte-identical to the
numpy path on blended rects, on the GUI thread and in a worker's chunk job.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from test_farm_terrain import FAKE_FARM_CONST, fake_farm  # noqa: F401 -- fake_farm is a fixture
from test_native_composite import (
    NEW_TERRAIN_ID,
    TERRAIN_IDS,
    _anchor_origins,
    _assert_backends_agree,
    _chunk_of_tile,
    _cliff_scenario,
    _corner_tiles,
    _count_native_calls,
    _patch_tiles,
    _rects,
    _sloped_scenario,
    native_kernel,  # noqa: F401 -- a fixture
    synthetic_textures,  # noqa: F401 -- a fixture
)
from test_sprite_edit_bbox import EDIT_X, EDIT_Y, sprite_install  # noqa: F401

from descape import asset_source, composite_backend, native_composite, render, terrain_blend
from descape.grid_overlay import grid_bake
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache, UnitSplice
from descape.terrain_blend import BlendEntry
from descape.view_layers import LayerState

LAYER_ID = 21  # not in TERRAIN_IDS
KINDS = ("flat", "stepped", "sloped", "sloped-flat-map")
BLEND_ON = LayerState()
BLEND_OFF = LayerState(terrain_blend=False)


def _fake(fn):
    fn.cache_clear = lambda: None
    return fn


@pytest.fixture
def blend_install(sprite_install, fake_farm, synthetic_textures, monkeypatch):  # noqa: F811
    """Random 0..128 alphas for every (mode, tile), a diagonal-ramp layer
    mask for every name, and a blend table over the fixtures' terrains in
    which NEW_TERRAIN_ID outranks everything, so painting it reaches all 8
    neighbours."""
    rng = np.random.default_rng(20261011)
    tiles = rng.integers(0, 129, size=(9, 31, asset_source.BLENDOMATIC_TILE_SIZE), dtype=np.uint8)
    size = asset_source.LOADED_TEXTURE_SIZE
    mask = (np.add.outer(np.arange(size), np.arange(size)) * 3 % 256).astype(np.uint8)
    ids = (*TERRAIN_IDS, LAYER_ID, NEW_TERRAIN_ID)
    table = {tid: BlendEntry(10 * (i + 1), i % 8, "") for i, tid in enumerate(ids)}
    monkeypatch.setattr(asset_source, "get_blendomatic", _fake(lambda: tiles))
    monkeypatch.setattr(asset_source, "get_terrain_mask_array", _fake(lambda _name: mask))
    monkeypatch.setattr(terrain_blend, "BLEND_TABLE", table)
    asset_source.clear_install_caches()
    assert render.blend_effective(LayerState())


@pytest.fixture(params=["numpy", "native"])
def backend(request):
    if request.param == "native" and not composite_backend.available():
        if os.environ.get("DESCAPE_REQUIRE_NATIVE") == "1":
            pytest.fail(f"DESCAPE_REQUIRE_NATIVE=1 but {composite_backend.unavailable_reason}")
        pytest.skip(composite_backend.unavailable_reason)
    with composite_backend.use_backend(request.param):
        yield request.param


def _with_layers(scenario):
    """Layers on about 1 tile in 6, some of them equal to the tile's own
    terrain (a no-op layer)."""
    for tile in scenario.map_manager.terrain:
        k = (tile.x * 5 + tile.y * 3) % 11
        if k == 0:
            tile.layer = LAYER_ID
        elif k == 1:
            tile.layer = tile.terrain_id
        elif k == 2:
            tile.layer = TERRAIN_IDS[(tile.x + tile.y) % len(TERRAIN_IDS)]
    return scenario


def _scenario(kind: str):
    scenario = _cliff_scenario() if kind == "stepped" else _sloped_scenario()
    if kind == "sloped-flat-map":
        for tile in scenario.map_manager.terrain:
            tile.elevation = 2
    return _with_layers(scenario)


def _style(kind: str) -> str:
    return kind.split("-", maxsplit=1)[0]


def _cache(kind: str, scenario, layers: LayerState = BLEND_ON, sprites: bool = False):
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    style = _style(kind)
    if style == "flat":
        return FlatChunkCache(scenario, tile_px, with_units=False, sprites=sprites, layers=layers)
    if style == "stepped":
        elevations, proj = render.elevations_and_proj(scenario)
        return IsoChunkCache(scenario, elevations, proj, tile_px, sprites=sprites, layers=layers)
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
    return SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=sprites, layers=layers)


def _full(kind: str, scenario, blend: bool = True, sprites: bool = False) -> np.ndarray:
    style = _style(kind)
    if style == "flat":
        return render.render_terrain(scenario, blend=blend)
    if style == "stepped":
        return render.render_terrain_iso_with_proj(scenario, with_sprites=sprites, blend=blend)[0]
    return render.render_terrain_sloped_with_proj(scenario, with_sprites=sprites, blend=blend)[0]


def _stitched(cache, mip: int = 0) -> np.ndarray:
    return cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)


def _assert_equal(got: np.ndarray, want: np.ndarray, what: str) -> None:
    want = want[: got.shape[0], : got.shape[1]]
    assert got.shape == want.shape, what
    if not np.array_equal(got, want):
        ys, xs = np.nonzero(np.any(got != want, axis=-1))
        pytest.fail(f"{what}: {ys.size} px differ, first at (y={ys[0]}, x={xs[0]}): {got[ys[0], xs[0]]} vs {want[ys[0], xs[0]]}")


def _keys(scenario, tiles) -> list:
    xs, ys = zip(*tiles, strict=True)
    return render._blend_keys(scenario, list(xs), list(ys))


@pytest.mark.parametrize("kind", KINDS)
def test_blending_engages_and_off_is_the_pre_blending_render(blend_install, monkeypatch, kind):
    """blend=False and the View > Layers row off both give exactly what a
    renderer with no blendomatic draws, and blending on changes pixels."""
    scenario = _scenario(kind)
    off = _full(kind, scenario, blend=False)
    assert not np.array_equal(_full(kind, scenario), off), "blending changed no pixel -- vacuous"
    _assert_equal(_stitched(_cache(kind, scenario, BLEND_OFF)), off, "row off vs blend=False")
    monkeypatch.setattr(asset_source, "get_blendomatic", _fake(lambda: None))
    asset_source.clear_install_caches()
    assert not render.blend_effective(LayerState())
    _assert_equal(_full(kind, scenario), off, "no blendomatic vs blend=False")


@pytest.mark.parametrize("kind", KINDS)
def test_textures_off_ignores_blending(blend_install, kind):
    """Blending draws over real textures only, so with textures off the row
    changes nothing."""
    scenario = _scenario(kind)
    textures_off = LayerState(terrain_textures=False)
    assert not render.blend_effective(textures_off)
    _assert_equal(
        _stitched(_cache(kind, scenario, textures_off)),
        _stitched(_cache(kind, scenario, LayerState(terrain_textures=False, terrain_blend=False))),
        "textures off, blending on vs off",
    )


@pytest.mark.parametrize("kind", KINDS)
def test_stitched_chunks_match_the_full_render(blend_install, backend, kind):
    scenario = _scenario(kind)
    _assert_equal(_stitched(_cache(kind, scenario)), _full(kind, scenario), f"{kind} on {backend}")


def test_stepped_equals_sloped_on_a_flat_map(blend_install):
    """The shared square top block makes this hold by construction."""
    scenario = _scenario("sloped-flat-map")
    stepped = render.render_terrain_iso(scenario)
    sloped = render.render_terrain_sloped(scenario)
    assert np.array_equal(stepped, sloped)
    assert not np.array_equal(stepped, render.render_terrain_iso(scenario, blend=False)), "vacuous"


def _patch_edit(kind: str, cache, scenario, x: int, y: int) -> None:
    """The viewer's own dirty funnel for a one-tile terrain edit."""
    if _style(kind) == "flat":
        mm = scenario.map_manager
        cache.patch_rects(render.flat_dirty_rects([(x, y)], mm.map_width, mm.map_height, cache.tile_px, cache.layers))
    else:
        _patch_tiles(cache, _style(kind), scenario, [(x, y)], sprites=False)


@pytest.mark.parametrize("edit", ["terrain", "layer"])
@pytest.mark.parametrize("kind", KINDS)
def test_a_one_tile_edit_patches_to_a_fresh_render(blend_install, backend, kind, edit):
    """One level warm, then one tile painted the top-priority terrain (all 8
    neighbours gain an influence from it, diagonals included) or given a
    layer (only its own key changes), patched through the viewer's funnel:
    the whole canvas equals a fresh cache. The edit leaves the cache's
    mirrors stale until the patch re-reads them. One level per cache, since
    showing another level evicts the first and would recomposite it fresh."""
    scenario = _scenario(kind)
    mm = scenario.map_manager
    x, y = EDIT_X + 1, EDIT_Y - 2
    tile = mm.terrain[y * mm.map_width + x]
    original = (tile.terrain_id, tile.layer)
    ring = [(x + dx, y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
    keys_before = _keys(scenario, [(x, y), *ring])

    def apply():
        if edit == "terrain":
            tile.terrain_id = NEW_TERRAIN_ID
        else:
            tile.layer = LAYER_ID

    assert original[0] != NEW_TERRAIN_ID and original[1] in (-1, original[0])
    apply()
    keys_after = _keys(scenario, [(x, y), *ring])
    moved = [k0 != k1 for k0, k1 in zip(keys_before, keys_after, strict=True)]
    # The top-priority centre keeps no key of its own; its crop changes instead.
    assert moved[1:] == [edit == "terrain"] * 8 and (edit == "terrain" or moved[0]), moved
    with composite_backend.use_backend("numpy"):
        fresh = _cache(kind, scenario)
        mips = [m for m in fresh.mip_levels() if m in (0, -1)]
        want = {m: _stitched(fresh, m).copy() for m in mips}

    for m in mips:
        tile.terrain_id, tile.layer = original
        cache = _cache(kind, scenario)
        before = _stitched(cache, m).copy()
        assert not np.array_equal(before, want[m]), f"mip {m}: the edit changed nothing -- vacuous"
        warm = {k: id(chunk) for k, chunk in cache._cache.items()}
        apply()
        _patch_edit(kind, cache, scenario, x, y)
        assert {k: id(chunk) for k, chunk in cache._cache.items()} == warm, f"mip {m}: the patch evicted -- vacuous"
        _assert_equal(_stitched(cache, m), want[m], f"{kind} mip {m} after a {edit} edit on {backend}")


@pytest.mark.parametrize("kind", ["stepped", "sloped"])
def test_a_farm_moved_onto_a_blended_boundary_matches_a_fresh_render(blend_install, backend, kind):
    """A farm-overridden tile takes the farm's plain crop and its neighbours
    keep keys read from the raw terrain_id, so a farm move through the
    splice path still equals a full render."""
    scenario = _scenario(kind)
    mm = scenario.map_manager
    farm = scenario.unit_manager.units[2][-1]
    assert farm.unit_const == FAKE_FARM_CONST
    cache = _cache(kind, scenario, sprites=True)
    _stitched(cache)
    nx, ny = 33.5, 41.5
    new_tiles = tuple(render.unit_occupied_tiles(_At(farm, nx, ny), mm.map_width, mm.map_height))
    assert any(k is not None for k in _keys(scenario, new_tiles)), "the farm lands on no blended tile -- vacuous"

    index = scenario.unit_manager.units[2].index(farm)
    old_own = (int(farm.x), int(farm.y))
    old_tiles = tuple(render.unit_occupied_tiles(farm, mm.map_width, mm.map_height))
    farm.x, farm.y = nx, ny
    scenario.unit_gen += 1
    splice = UnitSplice(2, index, farm, old_own, (int(nx), int(ny)), old_tiles, new_tiles)
    assert cache.can_splice([splice])
    cache.invalidate_units([splice])
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
    with composite_backend.use_backend("numpy"):
        want = _full(kind, scenario, sprites=True)
    _assert_equal(_stitched(cache), want, f"{kind} after a farm move on {backend}")


class _At:
    """`unit` as if it stood at (x, y), for unit_occupied_tiles() before the move."""

    def __init__(self, unit, x: float, y: float):
        self._unit, self.x, self.y = unit, x, y

    def __getattr__(self, name):
        return getattr(self._unit, name)


def test_flat_dirty_rects_dilate_only_while_blending_is_effective(blend_install):
    on = render.flat_dirty_rects([(0, 0), (1, 0)], 4, 4, 8, LayerState())
    assert sorted(on) == sorted((x * 8, y * 8, x * 8 + 8, y * 8 + 8) for x in range(3) for y in range(2))
    assert render.flat_dirty_rects([(0, 0)], 4, 4, 8, BLEND_OFF) == [(0, 0, 8, 8)]


def _blend_cache(kind: str, scenario):
    """Units, sprites, the farm and the grid all on, so the blended kernel
    path runs beside every other per-tile paint."""
    cache = _cache(kind, scenario, sprites=True)
    cache.set_grid(grid_bake(True, 60, 2))
    return cache


def _without_texture(monkeypatch, missing: int) -> None:
    real = asset_source.get_terrain_texture_array
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", _fake(lambda tid: None if tid == missing else real(tid)))
    asset_source.clear_install_caches()


@pytest.mark.parametrize("missing", [None, LAYER_ID, TERRAIN_IDS[2]], ids=["all", "no-layer-texture", "no-texture"])
@pytest.mark.parametrize("kind", ["stepped", "sloped"])
def test_blended_rects_backends_agree(native_kernel, blend_install, monkeypatch, kind, missing):  # noqa: F811
    """Every level's rects composite byte-identically on both backends, the
    native ones through the whole-rect kernel alone. A missing layer or
    neighbour texture is skipped by both, a missing own texture keeps its
    flat colour unblended."""
    if missing is not None:
        _without_texture(monkeypatch, missing)
    scenario = _scenario(kind)
    mm = scenario.map_manager
    cache = _blend_cache(kind, scenario)
    off = _blend_cache(kind, scenario)
    off.set_layers(BLEND_OFF)
    for mip in cache.mip_levels():
        lvl = cache._level(mip)
        heights = cache.elevations
        anchors = _anchor_origins(lvl.proj, heights, [(EDIT_X, EDIT_Y), *_corner_tiles(mm)])
        rects = _rects(cache, mip, anchors)
        with composite_backend.use_backend("native"):
            assert not np.array_equal(cache._composite_rect(mip, *rects[0]), off._composite_rect(mip, *rects[0])), (
                f"mip {mip}: blending changed nothing natively -- vacuous"
            )
        _assert_backends_agree(lambda *r, m=mip: cache._composite_rect(m, *r), rects)
        # The per-tile fallback blends too, so only the call count proves the kernel composed.
        calls = _count_native_calls(lambda *r, m=mip: cache._composite_rect(m, *r), cache.canvas_dims(mip))
        assert calls.keys() == {"composite_sloped" if kind == "sloped" else "composite_iso"}, (mip, calls)


@pytest.mark.parametrize("kind", ["stepped", "sloped"])
def test_a_blended_chunk_job_matches_numpy(native_kernel, blend_install, kind):  # noqa: F811
    """margin_warm's worker path: a chunk job prepared on this thread and run
    on another composes the same blended pixels as the numpy path."""
    import threading

    scenario = _scenario(kind)
    with composite_backend.use_backend("native"):
        cache = _blend_cache(kind, scenario)
        cache._level(0)
        key = _chunk_of_tile(cache, EDIT_X, EDIT_Y)
        job = cache.prepare_chunk_job(*key)
        assert job is not None, "no worker-safe path for a blended chunk"
        blend = job.args[-1]
        assert blend is not None and (blend[0] >= 0).any(), "the job blends no candidate -- vacuous"
        ran = []
        worker = threading.Thread(target=lambda: ran.append(job.run()))
        worker.start()
        worker.join()
        assert ran == [True], "the kernel declined"
    with composite_backend.use_backend("numpy"):
        want = _blend_cache(kind, scenario).get_chunk(*key)
    _assert_equal(job.scratch, want, f"{kind} chunk {key} from a worker job")


def test_a_mask_the_kernel_cannot_crop_falls_back_per_tile(native_kernel, blend_install, monkeypatch):  # noqa: F811
    """A layer mask larger than the textures still crops in numpy, but the
    kernel only takes the one size, so its blend tables refuse and every
    layered rect is composited per tile instead, with the same pixels."""
    size = 2 * asset_source.LOADED_TEXTURE_SIZE
    big = (np.add.outer(np.arange(size), np.arange(size)) * 5 % 256).astype(np.uint8)
    monkeypatch.setattr(asset_source, "get_terrain_mask_array", _fake(lambda _name: big))
    scenario = _scenario("stepped")
    cache = _blend_cache("stepped", scenario)
    calls = _count_native_calls(lambda *r: cache._composite_rect(0, *r), cache.canvas_dims())
    assert "render_tile_iso" in calls, calls
    anchors = _anchor_origins(cache._level(0).proj, cache.elevations, [(EDIT_X, EDIT_Y)])
    _assert_backends_agree(lambda *r: cache._composite_rect(0, *r), _rects(cache, 0, anchors))
    tids = np.array(TERRAIN_IDS)
    _slots, tex_tables, uniq = native_composite._textures(tids, 32, True, [LAYER_ID])
    key = (int(tids[0]), LAYER_ID, ())
    assert native_composite._blend_tables([key], tids[:1], None, 32, tex_tables, uniq) is native_composite.UNCOMPOSABLE
