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

import ast
import os
from pathlib import Path
from types import SimpleNamespace

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
from descape.brush import BRUSH_SHAPE_CIRCLE, brush_tiles
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


@pytest.fixture(autouse=True)
def blend_keys_match_the_mirrors(monkeypatch):
    """Every composite's blend keys, whichever entry point asked for them,
    equal influences() over the very mirrors it was handed. That makes each
    stroke, undo, patch, evict and farm-move case in a module using this
    fixture a check of the cache's key map at that state.

    Yields the counters: `composites` checked, and `derives`, the
    influences() calls the product path made (the check's own are not counted).
    A failed check is also kept on `failures` and re-raised at teardown, since
    MarginWarmer's tick catches Exception around its composites and only logs."""
    seen = SimpleNamespace(composites=0, derives=0, inside=False, failures=[])
    real_keys, real_influences = render._rect_blend_keys, terrain_blend.influences

    def influences(*args, **kwargs):
        seen.derives += seen.inside
        return real_influences(*args, **kwargs)

    def checked(scenario, candidates, layers, terrain_ids, layer_ids, key_map=None):
        seen.inside = True
        try:
            blend = real_keys(scenario, candidates, layers, terrain_ids, layer_ids, key_map)
        finally:
            seen.inside = False
        if candidates.shape[0] and render.blend_effective(layers):
            try:
                # influences() also raises on a candidate off the map, which the gather would wrap.
                want = render._blend_keys(scenario, candidates[:, 0], candidates[:, 1], terrain_ids, layer_ids)
                got = [None] * len(want) if blend is None else blend.as_list()
                assert got == want, "a composite's blend keys disagree with influences() over its own mirrors"
            except Exception as exc:
                seen.failures.append(exc)
                raise
            seen.composites += 1
        return blend

    monkeypatch.setattr(terrain_blend, "influences", influences)
    monkeypatch.setattr(render, "_rect_blend_keys", checked)
    yield seen
    assert not seen.failures, f"{len(seen.failures)} blend key check(s) failed, first: {seen.failures[0]!r}"


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
    """Blending on, the two tiles and their ring are one block rect, not six tile rects."""
    assert render.flat_dirty_rects([(0, 0), (1, 0)], 4, 4, 8, LayerState()) == [(0, 0, 24, 16)]
    assert render.flat_dirty_rects([(0, 0)], 4, 4, 8, BLEND_OFF) == [(0, 0, 8, 8)]


def _tile_sets() -> dict[str, list]:
    """Seeded tile sets on a 120x120 map for flat_tile_rects()' properties."""
    rng = np.random.default_rng(20261012)
    block = render.FLAT_PATCH_BLOCK_TILES
    full_block = [(2 * block + dx, block + dy) for dx in range(block) for dy in range(block)]
    scatter = [tuple(t) for t in rng.integers(0, 120, size=(40, 2)).tolist()]
    disc = brush_tiles(3 * block, 3 * block, 9, BRUSH_SHAPE_CIRCLE, 120, 120)
    edges = [(0, 0), (119, 0), (0, 119), (119, 119), *((x, 119) for x in range(100, 120)), *((119, y) for y in range(60, 70))]
    return {
        "full-block": full_block,
        "scatter": scatter,
        "disc-on-a-block-corner": disc,
        "map-edges": edges,
        "duplicates": [*disc, *disc[::2], *scatter[:5], *scatter[:5]],
        "one-tile": [(37, 81)],
        "two-far-in-a-block": [(block, block), (2 * block - 1, 2 * block - 1)],
    }


@pytest.mark.parametrize("name", list(_tile_sets()))
def test_flat_tile_rects_cover_their_tiles_in_disjoint_block_rects(name):
    """Every tile is covered, no two rects overlap, each rect lies inside one
    block, and a block's rects cover at most FLAT_PATCH_DENSITY times the
    tiles asked for there."""
    tiles = _tile_sets()[name]
    tile_px, block, density = 32, render.FLAT_PATCH_BLOCK_TILES, render.FLAT_PATCH_DENSITY
    rects = render.flat_tile_rects(iter(tiles), tile_px)
    assert rects == sorted(rects) and len(set(rects)) == len(rects)
    covered: dict[tuple[int, int], int] = {}
    per_block: dict[tuple[int, int], int] = {}
    for x0, y0, x1, y1 in rects:
        assert x0 < x1 and y0 < y1 and not any(v % tile_px for v in (x0, y0, x1, y1)), (x0, y0, x1, y1)
        tx0, ty0, tx1, ty1 = x0 // tile_px, y0 // tile_px, x1 // tile_px, y1 // tile_px
        key = (tx0 // block, ty0 // block)
        assert key == ((tx1 - 1) // block, (ty1 - 1) // block), f"rect {(tx0, ty0, tx1, ty1)} leaves its block"
        per_block[key] = per_block.get(key, 0) + (tx1 - tx0) * (ty1 - ty0)
        for x in range(tx0, tx1):
            for y in range(ty0, ty1):
                covered[(x, y)] = covered.get((x, y), 0) + 1
    assert set(tiles) <= covered.keys(), f"uncovered: {sorted(set(tiles) - covered.keys())[:5]}"
    assert max(covered.values()) == 1, "two rects overlap"
    asked: dict[tuple[int, int], int] = {}
    for x, y in set(tiles):
        asked[(x // block, y // block)] = asked.get((x // block, y // block), 0) + 1
    assert per_block.keys() == asked.keys()
    for key, n in asked.items():
        assert per_block[key] <= density * n, f"block {key}: {per_block[key]} tiles covered for {n} asked"
    if name == "full-block":
        assert rects == [(2 * block * tile_px, block * tile_px, 3 * block * tile_px, 2 * block * tile_px)]
    if name == "one-tile":
        assert rects == [(37 * tile_px, 81 * tile_px, 38 * tile_px, 82 * tile_px)]
    if name == "disc-on-a-block-corner":
        assert len(rects) == 4, "a disc across a block corner is one bbox per block"
    if name == "two-far-in-a-block":
        assert len(rects) == 2 and len(covered) == 2, "a sparse block falls back to one rect a tile"


def test_flat_tile_rects_of_nothing_is_empty():
    assert render.flat_tile_rects([], 32) == []


def _straddling_and_sparse_edit(monkeypatch) -> tuple[list, int]:
    """With 8-tile blocks: a 4x3 patch whose ring crosses a block edge, and
    two lone tiles whose rings each put one corner tile into block (2, 2),
    which is then sparse. Returns the tiles and the block side."""
    block = 8
    monkeypatch.setattr(render, "FLAT_PATCH_BLOCK_TILES", block)
    patch = [(EDIT_X + dx, EDIT_Y + dy) for dx in range(4) for dy in range(3)]
    return [*patch, (2 * block - 1, 2 * block - 1), (3 * block, 3 * block)], block


def test_a_multi_tile_flat_edit_patches_to_a_fresh_render(blend_install, backend, monkeypatch):
    """One level warm, then an edit whose rects take both flat_tile_rects()
    branches (block bboxes on either side of a block edge, and a sparse
    block's one-tile rects), patched through the viewer's funnel: the whole
    canvas equals a fresh cache, with no chunk evicted."""
    scenario = _scenario("flat")
    mm = scenario.map_manager
    tiles, block = _straddling_and_sparse_edit(monkeypatch)
    originals = {t: mm.terrain[t[1] * mm.map_width + t[0]].terrain_id for t in tiles}
    assert NEW_TERRAIN_ID not in originals.values()

    def apply():
        for x, y in tiles:
            mm.terrain[y * mm.map_width + x].terrain_id = NEW_TERRAIN_ID

    apply()
    with composite_backend.use_backend("numpy"):
        fresh = _cache("flat", scenario)
        mips = [m for m in fresh.mip_levels() if m in (0, -1)]
        want = {m: _stitched(fresh, m).copy() for m in mips}

    for m in mips:
        for (x, y), tid in originals.items():
            mm.terrain[y * mm.map_width + x].terrain_id = tid
        cache = _cache("flat", scenario)
        t = cache.tile_px
        before = _stitched(cache, m).copy()
        warm = {k: id(chunk) for k, chunk in cache._cache.items()}
        apply()
        rects = render.flat_dirty_rects(tiles, mm.map_width, mm.map_height, t, cache.layers)
        sizes = [((x1 - x0) // t, (y1 - y0) // t) for x0, y0, x1, y1 in rects]
        sparse = [(2 * block * t, 2 * block * t, (2 * block + 1) * t, (2 * block + 1) * t)]
        sparse.append(((3 * block - 1) * t, (3 * block - 1) * t, 3 * block * t, 3 * block * t))
        assert set(sparse) <= set(rects), "block (2, 2) did not fall back to one rect a tile -- vacuous"
        # The patch's ring is tiles 59..64 by 59..63: 5 columns left of the block edge at 64, 1 right of it.
        assert (5, 5) in sizes and (1, 5) in sizes, f"no bbox on both sides of a block edge -- vacuous: {sizes}"
        cache.patch_rects(rects)
        assert {k: id(chunk) for k, chunk in cache._cache.items()} == warm, f"mip {m}: the patch evicted -- vacuous"
        after = _stitched(cache, m)
        assert not np.array_equal(after, before), f"mip {m}: the edit changed nothing -- vacuous"
        _assert_equal(after, want[m], f"flat mip {m} after a multi-block edit on {backend}")


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


def _blended_skirts(composite, rects) -> int:
    """How many blended candidates with a left or right skirt (dl or dr > 0)
    the native composite_iso calls for `rects` were handed."""
    found = 0
    with composite_backend.use_backend("native"):
        real = composite_backend.native

        def recording(*args):
            nonlocal found
            cand_x, cand_y, elev, blend = args[3], args[4], args[7], args[13]
            if blend is not None:
                mh = elev.shape[0]
                for k in np.flatnonzero(blend[0] >= 0).tolist():
                    tx, ty = int(cand_x[k]), int(cand_y[k])
                    own = elev[ty, tx]
                    dl = own - elev[ty, tx - 1] if tx > 0 else 0
                    dr = own - elev[ty + 1, tx] if ty + 1 < mh else 0
                    found += dl > 0 or dr > 0
            return real.composite_iso(*args)

        class Recording:
            def __getattr__(self, name):
                return recording if name == "composite_iso" else getattr(real, name)

        composite_backend.native = Recording()  # use_backend restores it
        for rect in rects:
            composite(*rect)
    return found


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
        if kind == "stepped":
            assert _blended_skirts(lambda *r, m=mip: cache._composite_rect(m, *r), rects), (
                f"mip {mip}: no blended candidate has a skirt -- the skirt samples go untested"
            )
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
        blend = job.args[-2]  # two_pass follows it
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
    blend = _rect_keys([(int(tids[0]), LAYER_ID, ())])
    assert native_composite._blend_tables(blend, tids[:1], None, 32, tex_tables, uniq) is native_composite.UNCOMPOSABLE


def _rect_keys(keys: list) -> terrain_blend.RectKeys:
    """RectKeys over `keys` (None for no key), one candidate each."""
    key_map = terrain_blend.KeyMap(len(keys), 1)
    return terrain_blend.RectKeys(key_map.lookup(np.arange(len(keys)), lambda _xs, _ys: keys), key_map)


def test_blend_tables_build_one_spec_per_distinct_composing_key(blend_install):
    """Without a farm pack (overridden None): a keyless candidate, a key
    whose own tid is not the tid its crop reads, and two candidates sharing
    a key. Farm-overridden candidates drop out, and their tids still get
    texture slots."""
    a, b, c = TERRAIN_IDS[:3]
    shared = (a, LAYER_ID, ((b, 2, 4),))
    keys = [None, shared, (c, -1, ((b, 2, 5),)), shared, (b, -1, ((c, 2, 6),))]
    blend = _rect_keys(keys)
    assert blend.ids.tolist() == [terrain_blend.NO_KEY, 0, 1, 0, 2]
    base = np.array([a, a, a, a, b])  # candidate 2's key names c, its crop reads a
    assert native_composite._blend_extra_tids(blend) == sorted({LAYER_ID, b, c})
    assert native_composite._blend_extra_tids(None) == []
    _slots, tex_tables, uniq = native_composite._textures(base, 32, True, native_composite._blend_extra_tids(blend))
    slot = {int(t): i for i, t in enumerate(uniq.tolist())}
    cand, starts, ops, masks = native_composite._blend_tables(blend, base, None, 32, tex_tables, uniq)
    assert cand.tolist() == [-1, 0, -1, 0, 1] and cand.dtype == np.int32
    assert starts.tolist() == [0, 2, 3]
    kinds_slots = ops.reshape(-1, 3)[:, :2].tolist()
    assert kinds_slots == [
        [native_composite.BLEND_LAYER, slot[LAYER_ID]], [native_composite.BLEND_NEIGHBOUR, slot[b]],
        [native_composite.BLEND_NEIGHBOUR, slot[c]],
    ]
    assert len(masks) == 3

    farm = np.array([False, True, False, True, False])
    cand, starts, _ops, _masks = native_composite._blend_tables(blend, base, farm, 32, tex_tables, uniq)
    assert cand.tolist() == [-1, -1, -1, -1, 0] and starts.tolist() == [0, 1]
    assert native_composite._blend_tables(blend, base, np.ones(5, dtype=bool), 32, tex_tables, uniq) is None


# --- the cache's blend key map (TASK-031.63) ---


def _chunk_rects(cache, mip: int = 0) -> list:
    cw, ch = cache.canvas_dims(mip)
    p = cache.chunk_px
    return [cache._chunk_rect(mip, cx, cy) for cy in range(-(-ch // p)) for cx in range(-(-cw // p))]


@pytest.mark.parametrize("kind", KINDS)
def test_a_second_pass_derives_no_key_and_paints_the_same(blend_install, backend, kind, blend_keys_match_the_mirrors):
    """A level composited twice: the first pass fills the cache's key map,
    the second calls influences() zero times and returns the same pixels,
    and the stitched level still equals the full render. Fails if the map
    is rebound (so reset) on every call."""
    seen = blend_keys_match_the_mirrors
    scenario = _scenario(kind)
    cache = _cache(kind, scenario)
    rects = _chunk_rects(cache)
    first = [cache._composite_rect(0, *r) for r in rects]
    # An iso canvas corner rect can hold no tile at all, so not every rect is a checked composite.
    assert seen.derives > 0 and 0 < seen.composites <= len(rects), (seen.derives, seen.composites)
    key_map = cache._blend_key_map
    assert (key_map.ids >= 0).any() and not (key_map.ids == terrain_blend.UNKNOWN).any()
    seen.derives = 0
    second = [cache._composite_rect(0, *r) for r in rects]
    assert seen.derives == 0, f"{seen.derives} influences() calls on a warm key map"
    assert cache._blend_key_map is key_map
    for rect, a, b in zip(rects, first, second, strict=True):
        assert np.array_equal(a, b), rect
    _assert_equal(_stitched(cache), _full(kind, scenario), f"{kind} on {backend}, warm key map")


@pytest.mark.parametrize("kind", ["stepped", "sloped"])
def test_an_elevation_only_wholesale_refresh_keeps_every_key(blend_install, backend, kind, blend_keys_match_the_mirrors):
    """refresh_wholesale() re-reads both mirrors over the whole map; with no
    terrain id or layer changed it must invalidate no key, so the
    recomposite it forces derives nothing."""
    seen = blend_keys_match_the_mirrors
    scenario = _scenario(kind)
    mm = scenario.map_manager
    cache = _cache(kind, scenario)
    before = _stitched(cache).copy()
    key_map = cache._blend_key_map
    filled = key_map.ids != terrain_blend.UNKNOWN
    assert filled.all(), "the whole level was composited, so every tile should have been derived"

    dirty = [y * mm.map_width + x for x in range(EDIT_X - 2, EDIT_X + 3) for y in range(EDIT_Y - 2, EDIT_Y + 3)]
    for i in dirty:
        mm.terrain[i].elevation = (mm.terrain[i].elevation + 3) % 6
    assert cache.refresh_wholesale(dirty) is True
    assert cache._blend_key_map is key_map and not (key_map.ids == terrain_blend.UNKNOWN).any()
    seen.derives = seen.composites = 0
    after = _stitched(cache)
    assert seen.composites > 0 and seen.derives == 0, (seen.composites, seen.derives)
    assert not np.array_equal(after, before), "the elevation edit changed no pixel (vacuous)"
    with composite_backend.use_backend("numpy"):
        want = _stitched(_cache(kind, scenario))
    _assert_equal(after, want, f"{kind} after an elevation-only wholesale refresh on {backend}")


@pytest.mark.parametrize("kind", KINDS)
def test_a_rebound_blend_table_reaches_a_warm_cache(blend_install, backend, kind, monkeypatch):
    """The key map is bound to BLEND_TABLE by identity: with the table
    replaced and the chunks evicted WITHOUT a mirror re-read
    (terrain_changed=False), nothing but that binding can drop the old keys."""
    scenario = _scenario(kind)
    cache = _cache(kind, scenario)
    before = _stitched(cache).copy()
    table = terrain_blend.BLEND_TABLE
    top = max(e.priority for e in table.values())
    flipped = {tid: BlendEntry(top + 10 - e.priority, e.blend_type, e.overlay_mask) for tid, e in table.items()}
    monkeypatch.setattr(terrain_blend, "BLEND_TABLE", flipped)
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)), terrain_changed=False)
    after = _stitched(cache)
    assert not np.array_equal(after, before), "flipping every priority changed no pixel (vacuous)"
    _assert_equal(after, _full(kind, scenario), f"{kind} on {backend} after a BLEND_TABLE swap")


@pytest.mark.parametrize("kind", KINDS)
def test_replaced_mirrors_reach_a_warm_cache(blend_install, backend, kind):
    """The key map is bound to both mirror arrays by identity: a tile
    repainted, the mirrors replaced by fresh reads and the chunks evicted
    WITHOUT a re-read (terrain_changed=False), so no invalidate() runs and
    nothing but that binding can drop the old keys."""
    scenario = _scenario(kind)
    mm = scenario.map_manager
    cache = _cache(kind, scenario)
    before = _stitched(cache).copy()
    key_map = cache._blend_key_map
    mm.terrain[EDIT_Y * mm.map_width + EDIT_X].terrain_id = NEW_TERRAIN_ID
    cache._terrain_ids = render.tile_field_mirror(scenario, "terrain_id")
    cache._layer_ids = render.tile_field_mirror(scenario, "layer")
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)), terrain_changed=False)
    after = _stitched(cache)
    assert cache._blend_key_map is key_map, "the cache made a new map, so the binding was not what dropped the keys"
    assert not np.array_equal(after, before), "the repaint changed no pixel (vacuous)"
    _assert_equal(after, _full(kind, scenario), f"{kind} on {backend} after both mirrors were replaced")


@pytest.mark.parametrize("kind", KINDS)
def test_a_rebound_derive_function_reaches_a_warm_cache(blend_install, backend, kind, monkeypatch):
    """The key map is bound to render._blend_keys by identity too (the
    review pack's one-tile-plain inject rebinds it): with a derive that
    gives no tile a key, a warm cache repaints unblended like the full
    render, not from the keys it already held."""
    scenario = _scenario(kind)
    cache = _cache(kind, scenario)
    before = _stitched(cache).copy()
    monkeypatch.setattr(render, "_blend_keys", lambda _scenario, xs, _ys, *_mirrors: [None] * len(xs))
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)), terrain_changed=False)
    after = _stitched(cache)
    assert not np.array_equal(after, before), "dropping every key changed no pixel (vacuous)"
    _assert_equal(after, _full(kind, scenario), f"{kind} on {backend} after _blend_keys was rebound")


@pytest.mark.parametrize("kind", KINDS)
def test_a_cache_whose_key_map_keeps_resetting_paints_the_same(blend_install, backend, kind, monkeypatch):
    """Past KEY_MAP_MAX_SPECS the map starts over inside lookup(). With the
    cap at 4 a level's composites cross it again and again, and the stitched
    level still equals the full render: no composite reads an id from a
    table that was reset under it."""
    resets = []
    real_reset = terrain_blend.KeyMap.reset
    monkeypatch.setattr(terrain_blend.KeyMap, "reset", lambda self: (resets.append(1), real_reset(self))[1])
    monkeypatch.setattr(terrain_blend, "KEY_MAP_MAX_SPECS", 4)
    scenario = _scenario(kind)
    cache = _cache(kind, scenario)
    resets.clear()
    got = _stitched(cache)
    assert len(resets) >= 2, f"{len(resets)} reset(s): the cap was not crossed repeatedly (vacuous)"
    _assert_equal(got, _full(kind, scenario), f"{kind} on {backend} with the key map resetting")


def test_a_flat_patch_derives_its_keys_in_one_call(blend_install, backend, blend_keys_match_the_mirrors, monkeypatch):
    """Flat's sparse fallback patches one tile rect at a time (every block
    here, with the density at 0); the keys an edit invalidated are derived
    once for all of them, not once per rect, and the patched canvas equals
    a fresh cache."""
    seen = blend_keys_match_the_mirrors
    monkeypatch.setattr(render, "FLAT_PATCH_DENSITY", 0)
    scenario = _scenario("flat")
    mm = scenario.map_manager
    cache = _cache("flat", scenario)
    before = _stitched(cache).copy()
    tiles = [(EDIT_X + dx, EDIT_Y + dy) for dx in range(4) for dy in range(3)]
    for x, y in tiles:
        mm.terrain[y * mm.map_width + x].terrain_id = NEW_TERRAIN_ID
    rects = render.flat_dirty_rects(tiles, mm.map_width, mm.map_height, cache.tile_px, cache.layers)
    assert len(rects) == 6 * 5, "the dirty set was not dilated by the ring"
    seen.derives = seen.composites = 0
    cache.patch_rects(rects)
    assert seen.derives == 1 and seen.composites == len(rects) + 1, (seen.derives, seen.composites)
    after = _stitched(cache)
    assert not np.array_equal(after, before), "the edit changed no pixel (vacuous)"
    with composite_backend.use_backend("numpy"):
        want = _stitched(_cache("flat", scenario))
    _assert_equal(after, want, f"flat after a 12-tile edit on {backend}")


def _chunks_under(cache, rect, mip: int = 0) -> int:
    """Cached chunks of level `mip` the REFERENCE rect `rect` reaches."""
    cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, *cache._bbox_to_level(mip, rect))
    return sum(cache.has_chunk(mip, cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1))


def _count_composites(cache, monkeypatch) -> list:
    """Every _composite_rect() call from here on, through a fake that delegates."""
    calls = []
    real = cache._composite_rect

    def counting(mip, x0, y0, x1, y1):
        calls.append((mip, x0, y0, x1, y1))
        return real(mip, x0, y0, x1, y1)

    monkeypatch.setattr(cache, "_composite_rect", counting)
    return calls


def test_a_flat_edit_inside_one_block_composites_once_per_chunk(blend_install, backend, monkeypatch):
    """One resident level and a 6x6 patch whose ring stays inside one block
    and crosses the fixture's 8-tile chunk edge both ways: _composite_rect()
    runs once per chunk the one rect reaches (4), not once per dirty tile
    and neighbour (64)."""
    scenario = _scenario("flat")
    mm = scenario.map_manager
    cache = _cache("flat", scenario)
    before = _stitched(cache).copy()
    t = cache.tile_px
    assert cache.resident_levels() == [0] and cache.chunk_px == 8 * t
    block = render.FLAT_PATCH_BLOCK_TILES
    x0 = y0 = 3 * block + 4
    tiles = [(x0 + dx, y0 + dy) for dx in range(6) for dy in range(6)]
    for x, y in tiles:
        mm.terrain[y * mm.map_width + x].terrain_id = NEW_TERRAIN_ID
    rects = render.flat_dirty_rects(tiles, mm.map_width, mm.map_height, t, cache.layers)
    calls = _count_composites(cache, monkeypatch)
    cache.patch_rects(rects)
    composites = len(calls)
    assert composites == 4, f"{composites} composites for a ring of 64 tiles over 4 chunks"
    assert rects == [((x0 - 1) * t, (y0 - 1) * t, (x0 + 7) * t, (y0 + 7) * t)]
    assert _chunks_under(cache, rects[0]) == 4
    after = _stitched(cache)
    assert not np.array_equal(after, before), "the edit changed no pixel (vacuous)"
    with composite_backend.use_backend("numpy"):
        want = _stitched(_cache("flat", scenario))
    _assert_equal(after, want, f"flat after a one-block edit on {backend}")


def test_a_multi_block_flat_edit_derives_no_tile_key_twice(blend_install, backend, blend_keys_match_the_mirrors, monkeypatch):
    """A brush-9 disc across a block corner: one rect per block, one
    _composite_rect() per chunk each reaches, and no tile index is handed
    to influences() twice in the whole patch."""
    seen = blend_keys_match_the_mirrors
    scenario = _scenario("flat")
    mm = scenario.map_manager
    w = mm.map_width
    cache = _cache("flat", scenario)
    before = _stitched(cache).copy()
    assert cache.resident_levels() == [0]
    block = render.FLAT_PATCH_BLOCK_TILES
    tiles = brush_tiles(3 * block, 3 * block, 9, BRUSH_SHAPE_CIRCLE, w, mm.map_height)
    for x, y in tiles:
        mm.terrain[y * w + x].terrain_id = NEW_TERRAIN_ID
    rects = render.flat_dirty_rects(tiles, w, mm.map_height, cache.tile_px, cache.layers)
    derived: list[int] = []
    counted = terrain_blend.influences  # the autouse fixture's fake, which delegates

    def recording(tid_mirror, layer_mirror, xs, ys):
        if seen.inside:
            derived.extend((np.asarray(ys) * w + np.asarray(xs)).tolist())
        return counted(tid_mirror, layer_mirror, xs, ys)

    monkeypatch.setattr(terrain_blend, "influences", recording)
    calls = _count_composites(cache, monkeypatch)
    cache.patch_rects(rects)
    composites = len(calls)
    assert len(rects) == 4, f"{len(rects)} rects for a disc whose ring reaches four blocks"
    assert composites == sum(_chunks_under(cache, rect) for rect in rects)
    assert derived, "the patch derived no key (vacuous)"
    assert len(derived) == len(set(derived)), "a tile's blend key was derived twice in one patch"
    after = _stitched(cache)
    assert not np.array_equal(after, before), "the edit changed no pixel (vacuous)"
    with composite_backend.use_backend("numpy"):
        want = _stitched(_cache("flat", scenario))
    _assert_equal(after, want, f"flat after a four-block edit on {backend}")


def test_a_flat_patch_of_one_level_evicts_the_others_under_its_rects(blend_install, backend):
    """TASK-338: patch_rects(rects, levels=(0,)) with mip 0 and mip -1 warm,
    for a block rect and a sparse scatter of one-tile rects. Every mip 0
    chunk stays the same object and the level equals a fresh cache; at mip
    -1 the chunks under a rect are gone, every other one (the one between
    the two groups included) is the same object, the byte count matches,
    and the level read back equals a fresh cache."""
    scenario = _scenario("flat")
    mm = scenario.map_manager
    cache = _cache("flat", scenario)
    cache.max_bytes *= 4  # room for both levels at once
    t = cache.tile_px
    assert cache.chunk_px == 8 * t and cache.mip_tile_px(-1) * 2 == t, "mip -1 chunks are not 16 tiles"
    before = {mip: _stitched(cache, mip).copy() for mip in (0, -1)}
    assert cache.resident_levels() == [-1, 0]
    # A 5x5 block in mip -1 chunk (1, 1), and two lone tiles in chunk (5, 5); chunk (3, 3) lies between.
    tiles = [(x, y) for x in range(20, 25) for y in range(20, 25)] + [(81, 81), (93, 93)]
    for x, y in tiles:
        mm.terrain[y * mm.map_width + x].terrain_id = NEW_TERRAIN_ID
    rects = render.flat_dirty_rects(tiles, mm.map_width, mm.map_height, t, cache.layers)
    block = [r for r in rects if r[0] < 40 * t]
    sparse = [r for r in rects if r[0] >= 40 * t]
    assert block == [(19 * t, 19 * t, 26 * t, 26 * t)], block
    assert len(sparse) == 18 and all(r[2] - r[0] == t and r[3] - r[1] == t for r in sparse), sparse
    under = {(-1, 1, 1), (-1, 5, 5)}
    assert under <= cache._cache.keys() and (-1, 3, 3) in cache._cache
    kept = {key: chunk for key, chunk in cache._cache.items() if key not in under}

    cache.patch_rects(rects, levels=(0,))

    # Read before the levels are read back, which refills them; asserted after the pixels.
    stale = under & cache._cache.keys()
    keys_kept = cache._cache.keys() - under == kept.keys()
    same_objects = all(cache._cache.get(key) is chunk for key, chunk in kept.items())
    bytes_right = cache._cache_bytes == sum(chunk.nbytes for chunk in cache._cache.values())
    with composite_backend.use_backend("numpy"):
        fresh = _cache("flat", scenario)
        fresh.max_bytes *= 4
        want = {mip: _stitched(fresh, mip) for mip in (0, -1)}
    after = {mip: _stitched(cache, mip) for mip in (0, -1)}
    for mip in (0, -1):
        _assert_equal(after[mip], want[mip], f"mip {mip} after a one-level patch on {backend}")
        assert not np.array_equal(after[mip], before[mip]), f"mip {mip}: the edit changed no pixel (vacuous)"
    assert not stale, f"mip -1 chunks under a rect still cached: {sorted(stale)}"
    assert keys_kept, "a chunk outside the rects' mip -1 chunks was dropped"
    assert same_objects, "a kept chunk was replaced"
    assert bytes_right, "_cache_bytes does not match the cached chunks"


MIRRORS = ("_terrain_ids", "_layer_ids")
# The names the same arrays travel under once a cache has passed them on.
MIRROR_NAMES = ("terrain_ids", "layer_ids", "tid_mirror", "layer_mirror")
IN_PLACE_METHODS = ("fill", "put", "itemset", "sort", "partition", "resize", "setfield", "byteswap")
IN_PLACE_FUNCTIONS = ("put", "copyto", "place", "putmask", "put_along_axis")


def _in_place_mirror_writes() -> list[tuple[str, str, str]]:
    """(file, enclosing function, mirror) for every in-place write to a
    chunk cache's terrain mirrors under descape/, as an attribute or under
    a name they are passed on as: a subscript assignment, an augmented
    assignment, a mutating ndarray method, or a numpy function that writes
    its first argument."""
    found = []
    for path in sorted((Path(render.__file__).parent).glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def owner(node, parents=parents) -> str:
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
            return "<module>"

        def mirror(node) -> str | None:
            if isinstance(node, ast.Attribute) and node.attr in MIRRORS:
                return node.attr
            return node.id if isinstance(node, ast.Name) and node.id in MIRROR_NAMES else None

        for node in ast.walk(tree):
            names = []
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    names.extend(mirror(sub.value) for sub in ast.walk(target) if isinstance(sub, ast.Subscript))
                if isinstance(node, ast.AugAssign):
                    names.append(mirror(node.target))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in IN_PLACE_METHODS:
                    names.append(mirror(node.func.value))
                if node.func.attr in IN_PLACE_FUNCTIONS and node.args:
                    names.append(mirror(node.args[0]))
            found.extend((path.name, owner(node), name) for name in names if name)
    return found


def test_the_terrain_mirrors_have_one_in_place_writer():
    """The key map is invalidated where the mirrors are written, which is
    sound only while _refresh_terrain_ids() is the one place that writes
    them in place. A new writer must invalidate the map itself, and then
    belongs in this pin."""
    assert sorted(_in_place_mirror_writes()) == [
        ("render_cache.py", "_refresh_terrain_ids", "_layer_ids"),
        ("render_cache.py", "_refresh_terrain_ids", "_terrain_ids"),
    ]
