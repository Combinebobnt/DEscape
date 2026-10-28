"""The coarsest level's box-reduced store (TASK-031.70): render_cache.box_reduce2()
and render_reduced_rect(), the cached copy MapCanvasItem.paint() blits zoomed
out past the coarsest level.

The oracle is always a FRESH cache: box_reduce2() applied k times to its full
coarsest-level render_rect(). A reduced entry is valid only while every real
coarsest-level chunk under it is cached with the stamp it had when the entry
was built, so each viewer edit runs here as its real call sequence on a cache
that already holds entries at k = 1, 2 and 3, read back k = 3 first so a stale
k-1 entry is still there when k is checked. Each edit case first asserts that
the oracle changed, so a stale entry cannot pass by matching.

units_120x120 with synthetic textures and a terrain patchwork: under pytest
the real textures are hidden and a blank map renders nearly flat, where an
edit may change no pixel at all."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from test_native_composite import TERRAIN_IDS, native_kernel, synthetic_textures  # noqa: F401 -- fixtures
from test_pooled_render_rect import WORKERS, pool  # noqa: F401 -- a fixture
from test_sprite_edit_bbox import Unit, sprite_install  # noqa: F401 -- a fixture

from descape import composite_backend, render, render_cache
from descape.grid_overlay import grid_bake
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache, UnitSplice, _ChunkCacheBase
from descape.scenario_io import load_map_and_units
from descape.unit_filter import UnitFilter
from testkit import qt_capture

import conftest
from test_unit_sprites import CONST

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
STYLES = ("stepped", "sloped", "flat")
KS = (3, 2, 1)
# The sprite unit, and the terrain band every terrain edit repaints: 40 x 8 tiles, which
# at the coarsest level (tile_px 16) survives 8x8 averaging with room to spare.
UX, UY = 60, 60
BAND = (30, 70, 50, 58)
NEW_TERRAIN_ID = 23


@pytest.fixture
def env(synthetic_textures, sprite_install):  # noqa: F811
    """Synthetic textures and one installed sprite (CONST) for every case."""


def _reference_reduce(src: np.ndarray) -> np.ndarray:
    """The Decisions filter written out pixel by pixel."""
    h, w, c = src.shape
    out = np.zeros(((h + 1) // 2, (w + 1) // 2, c), dtype=np.uint8)
    for i in range(out.shape[0]):
        ys = (2 * i, min(2 * i + 1, h - 1))
        for j in range(out.shape[1]):
            xs = (2 * j, min(2 * j + 1, w - 1))
            for ch in range(c):
                out[i, j, ch] = (sum(int(src[y, x, ch]) for y in ys for x in xs) + 2) >> 2
    return out


def _scenario():
    scenario = load_map_and_units(str(FIXTURE))
    for tile in scenario.map_manager.terrain:
        tile.terrain_id = TERRAIN_IDS[(tile.x // 3 + tile.y // 5) % len(TERRAIN_IDS)]
    # Centred on its tile: Flat patches a unit's footprint tiles, which a corner-anchored icon overhangs.
    scenario.unit_manager.units[1].append(Unit(UX + 0.5, UY + 0.5, CONST))
    return scenario


def _make(style: str, scenario, **kw):
    tile_px = render.tile_pixels_for_map(scenario.map_manager.map_width, scenario.map_manager.map_height)
    if style == "stepped":
        elevations, proj = render.elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=False, **kw)
    elif style == "sloped":
        elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=False, **kw)
    else:
        cache = FlatChunkCache(scenario, tile_px, sprites=False, **kw)
    cache.set_grid(grid_bake(True, 60, 2))
    return cache


def _fresh_like(style: str, scenario, cache):
    """A new cache over the scenario as it is now, with `cache`'s view state."""
    fresh = _make(style, scenario)
    fresh.set_unit_filter(cache.unit_filter)
    fresh.set_layers(cache.layers)
    fresh.set_grid(cache.grid)
    if cache.sprites_enabled:
        fresh.set_sprites_enabled(True)
    return fresh


def _coarsest(cache) -> int:
    mip = cache.mip_levels()[0]
    assert mip != 0, "the fixture needs a coarser level than 0"
    return mip


def _oracle(cache, k: int) -> np.ndarray:
    mip = _coarsest(cache)
    img = cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)
    for _ in range(k):
        img = render_cache.box_reduce2(img)
    return img


def _reduced(cache, k: int) -> np.ndarray:
    mip = _coarsest(cache)
    return cache.render_reduced_rect(0, 0, *cache.reduced_dims(mip, k), mip, k)


def _differs(got: np.ndarray, want: np.ndarray) -> bool:
    return got.shape != want.shape or not np.array_equal(got, want)


def _assert_same(got: np.ndarray, want: np.ndarray, what: str) -> None:
    assert want.any(), f"{what}: rendered all black -- proves nothing"
    assert got.shape == want.shape, f"{what}: {got.shape} vs {want.shape}"
    if not np.array_equal(got, want):
        ys, xs = np.nonzero(np.any(got != want, axis=-1))
        pytest.fail(f"{what}: {ys.size} px differ, first at (y={ys[0]}, x={xs[0]})")


# --- the edits, each its real call sequence (viewer.py's unit and terrain funnels) ----------


def _band_tiles():
    x0, x1, y0, y1 = BAND
    return [(x, y) for y in range(y0, y1) for x in range(x0, x1)]


def _paint_band(scenario) -> list[tuple[int, int]]:
    mm = scenario.map_manager
    tiles = _band_tiles()
    for x, y in tiles:
        mm.terrain[y * mm.map_width + x].terrain_id = NEW_TERRAIN_ID
    return tiles


def _dirty(style: str, scenario, cache, tiles, with_units: bool = True):
    """(bbox, elevation_changed) the viewer patches for `tiles` (REFERENCE pixels).
    Stepped/Sloped through the dirty bbox, which syncs cache.elevations; Flat the
    tiles' rect grown by a tile, which covers a blend."""
    w = scenario.map_manager.map_width
    if style == "flat":
        t = cache.tile_px
        xs, ys = [x for x, _y in tiles], [y for _x, y in tiles]
        return ((min(xs) - 1) * t, (min(ys) - 1) * t, (max(xs) + 2) * t, (max(ys) + 2) * t), set()
    changed: set = set()
    fn = render.dirty_screen_bbox_iso if style == "stepped" else render.dirty_screen_bbox_sloped
    bbox = fn(
        scenario, [y * w + x for x, y in tiles], cache.elevations, cache.proj, with_units=with_units,
        with_sprites=cache.sprites_enabled, elevation_changed=changed,
    )
    assert bbox is not None
    return bbox, changed


def _op_patch(style, scenario, cache):
    bbox, changed = _dirty(style, scenario, cache, _paint_band(scenario))
    cache.patch(bbox, elevation_changed=changed)


def _op_patch_rects(style, scenario, cache):
    tiles = _paint_band(scenario)
    if style == "flat":
        mm = scenario.map_manager
        cache.patch_rects(render.flat_dirty_rects(tiles, mm.map_width, mm.map_height, cache.tile_px, cache.layers))
        return
    x0, y0, x1, y1 = _dirty(style, scenario, cache, tiles)[0]
    mid = (x0 + x1) // 2
    cache.patch_rects([(x0, y0, mid, y1), (mid, y0, x1, y1)])


def _op_patch_sources_composite(style, scenario, cache):
    bbox, changed = _dirty(style, scenario, cache, _paint_band(scenario))
    if cache.patch_sources(bbox, changed):
        cache.patch_composite(bbox)


def _move_units(scenario, cache) -> list[tuple[int, int]]:
    """Every player unit 9 tiles along x, through invalidate_units(); returns the tiles touched."""
    mm = scenario.map_manager
    splices, touched = [], []
    for player_id in range(1, len(scenario.unit_manager.units)):
        for index, unit in enumerate(scenario.unit_manager.units[player_id]):
            old_tiles = tuple(render.unit_occupied_tiles(unit, mm.map_width, mm.map_height) or ())
            old_own = (int(unit.x), int(unit.y))
            unit.x = unit.x + 9 if unit.x < mm.map_width - 12 else unit.x - 9
            new_tiles = tuple(render.unit_occupied_tiles(unit, mm.map_width, mm.map_height) or ())
            splices.append(UnitSplice(player_id, index, unit, old_own, (int(unit.x), int(unit.y)), old_tiles, new_tiles))
            touched += [*old_tiles, *new_tiles]
    assert len(splices) >= 3, "the fixture's units moved nothing"
    scenario.unit_gen += 1
    cache.invalidate_units(splices)
    return touched


def _op_units_patch(style, scenario, cache):
    touched = _move_units(scenario, cache)
    if style == "flat":
        cache.patch_rects(render.flat_tile_rects(touched, cache.tile_px))
        return
    bbox, changed = _dirty(style, scenario, cache, touched)
    cache.patch(bbox, elevation_changed=changed)


def _op_units_patch_composite_rects(style, scenario, cache):
    touched = _move_units(scenario, cache)
    if style == "flat":
        cache.patch_composite_rects(render.flat_tile_rects(touched, cache.tile_px))
        return
    bbox, _changed = _dirty(style, scenario, cache, touched)
    cache.patch_composite_rects(cache.chunk_rects([bbox], _coarsest(cache)))


def _op_invalidate_region(style, scenario, cache):
    bbox, _changed = _dirty(style, scenario, cache, _paint_band(scenario))
    cache.invalidate_region(bbox)


def _op_set_unit_filter(style, scenario, cache):
    cache.set_unit_filter(UnitFilter(players=frozenset(), show_gaia=False))


def _op_set_layers(style, scenario, cache):
    cache.set_layers(dataclasses.replace(cache.layers, terrain_textures=False))


def _op_set_grid(style, scenario, cache):
    cache.set_grid(grid_bake(False, 60, 2))


def _op_sprites(style, scenario, cache):
    cache.set_sprites_enabled(True)


def _op_refresh_wholesale(style, scenario, cache):
    """A one-level plateau (Sloped's legal shape), applied wholesale as the
    viewer does past STEPPED_FULL_RERENDER_THRESHOLD."""
    mm = scenario.map_manager
    dirty = []
    for x, y in _band_tiles():
        tile = mm.terrain[y * mm.map_width + x]
        tile.elevation += 1
        dirty.append(y * mm.map_width + x)
    assert cache.refresh_wholesale(dirty) is True


OPS = {
    "patch": _op_patch,
    "patch_rects": _op_patch_rects,
    "patch_sources+patch_composite": _op_patch_sources_composite,
    "invalidate_units+patch": _op_units_patch,
    "invalidate_units+patch_composite_rects": _op_units_patch_composite_rects,
    "invalidate_region": _op_invalidate_region,
    "set_unit_filter": _op_set_unit_filter,
    "set_layers": _op_set_layers,
    "set_grid": _op_set_grid,
    "sprites": _op_sprites,
    "refresh_wholesale": _op_refresh_wholesale,
}
CASES = [(style, op) for style in STYLES for op in OPS if not (style == "flat" and op == "refresh_wholesale")]


def _stale_after(style: str, op: str, cache=None, scenario=None) -> list[int]:
    """Builds reduced entries at every k, runs `op`, and returns the k whose
    reduced render differs from the fresh oracle (k = 3 read first)."""
    scenario = scenario or _scenario()
    cache = cache or _make(style, scenario)
    before = {k: _reduced(cache, k) for k in KS}
    OPS[op](style, scenario, cache)
    fresh = _fresh_like(style, scenario, cache)
    stale = []
    for k in KS:
        want = _oracle(fresh, k)
        assert want.any(), f"{style} {op} k={k}: the oracle is all black"
        assert _differs(want, before[k]), f"{style} {op} k={k}: the edit changed nothing the oracle sees"
        if _differs(_reduced(cache, k), want):
            stale.append(k)
    return stale


# --- box_reduce2 -------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "native"])
@pytest.mark.parametrize("shape", [(8, 6), (9, 7), (1, 1), (1, 5), (5, 1), (2, 3)])
def test_box_reduce2_matches_a_pure_python_reference(backend, shape):
    if backend == "native" and not composite_backend.available():
        pytest.skip(composite_backend.unavailable_reason)
    rng = np.random.default_rng(sum(shape))
    cases = [
        rng.integers(0, 256, (*shape, 3), dtype=np.uint8),
        np.full((*shape, 3), 255, dtype=np.uint8),  # 4 * 255 + 2 must not wrap
        np.zeros((*shape, 3), dtype=np.uint8),
    ]
    alternating = np.zeros((*shape, 3), dtype=np.uint8)
    alternating[::2] = 255
    cases.append(alternating)  # a pair of 0 and 255 rounds 127.5 and 191.25 the same way every time
    with composite_backend.use_backend(backend):
        for src in cases:
            np.testing.assert_array_equal(render_cache.box_reduce2(src), _reference_reduce(src))


def test_reduced_dims_round_up():
    cache = _make("stepped", _scenario())
    mip = _coarsest(cache)
    w, h = cache.canvas_dims(mip)
    for k in (1, 2, 3):
        assert cache.reduced_dims(mip, k) == (-(-w // 2**k), -(-h // 2**k))


# --- render_reduced_rect: identity with the fresh oracle ---------------------


@pytest.mark.parametrize("style", STYLES)
def test_reduced_render_is_box_reduce_k_times_of_a_fresh_full_render(env, style):
    scenario = _scenario()
    cache = _make(style, scenario)
    fresh = _make(style, scenario)
    mip = _coarsest(cache)
    for k in (1, 2, 3):
        want = _oracle(fresh, k)
        _assert_same(_reduced(cache, k), want, f"{style} k={k} whole canvas")
        rw, rh = cache.reduced_dims(mip, k)
        for x0, y0, x1, y1 in ((1, 3, rw // 2 + 5, rh // 3 + 7), (rw // 3 - 1, rh // 2 + 1, rw, rh), (7, 0, 9, rh)):
            got = cache.render_reduced_rect(x0, y0, x1, y1, mip, k)
            _assert_same(got, want[y0:y1, x0:x1], f"{style} k={k} rect {(x0, y0, x1, y1)}")


@pytest.mark.parametrize("style", STYLES)
def test_a_reduced_read_out_of_order_and_twice_is_the_same(env, style):
    """Sub-rects first (building k-1 entries piecemeal), then hits."""
    scenario = _scenario()
    cache = _make(style, scenario)
    mip = _coarsest(cache)
    rw, rh = cache.reduced_dims(mip, 1)
    cache.render_reduced_rect(rw // 2, rh // 2, rw, rh, mip, 1)
    want = _oracle(_make(style, scenario), 3)
    _assert_same(_reduced(cache, 3), want, f"{style} k=3 after a partial k=1 read")
    _assert_same(_reduced(cache, 3), want, f"{style} k=3 hit")


# --- coherence: one case per edit ----------------------------------------------


@pytest.mark.parametrize(("style", "op"), CASES)
def test_reduced_entries_follow_every_edit(env, style, op):
    assert _stale_after(style, op) == [], f"{style}: a stale reduced entry served after {op}"


@pytest.mark.parametrize("style", STYLES)
def test_an_evicted_then_edited_parent_rebuilds_its_reduced_entries(env, style):
    """patch() skips an uncached chunk, so an entry over an evicted parent may
    be stale even though no stamp moved: the presence check is what refuses it."""
    assert _evicted_then_edited_stale(style) == []


def _evicted_then_edited_stale(style: str) -> list[int]:
    scenario = _scenario()
    cache = _make(style, scenario)
    mip = _coarsest(cache)
    before = {k: _reduced(cache, k) for k in KS}
    for key in [key for key in cache._cache if key[0] == mip]:
        cache._cache_bytes -= cache._cache.pop(key).nbytes
    bbox, changed = _dirty(style, scenario, cache, _paint_band(scenario))
    cache.patch(bbox, elevation_changed=changed)
    fresh = _fresh_like(style, scenario, cache)
    stale = []
    for k in KS:
        want = _oracle(fresh, k)
        assert _differs(want, before[k])
        if _differs(_reduced(cache, k), want):
            stale.append(k)
    return stale


@pytest.mark.parametrize("style", STYLES)
def test_a_hit_survives_its_k_minus_1_entries_being_evicted(env, style, monkeypatch):
    scenario = _scenario()
    cache = _make(style, scenario)
    mip = _coarsest(cache)
    _reduced(cache, 2)
    for key in [key for key in cache._reduced if key[1] == 1]:
        cache._reduced_bytes -= cache._reduced.pop(key).pixels.nbytes
    builds = _count_builds(monkeypatch)
    got = _reduced(cache, 2)
    assert builds == [], "a valid k=2 entry rebuilt after its k=1 entries left"
    _assert_same(got, _oracle(_make(style, scenario), 2), f"{style} k=2")
    assert all(key[1] == 2 for key in cache._reduced if key[0] == mip)


def _count_builds(monkeypatch) -> list:
    builds: list = []
    real = render_cache.box_reduce2

    def spy(src):
        builds.append(src.shape)
        return real(src)

    monkeypatch.setattr(render_cache, "box_reduce2", spy)
    return builds


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_pooled_paths_keep_reduced_entries_coherent(env, native_kernel, pool, style, monkeypatch):  # noqa: F811
    """A cold reduced build composites its parents through _pooled_chunks(), and
    an edit's slices through _pooled_patch(); both must stamp what they write.
    On the native backend even under DESCAPE_COMPOSITE=numpy: numpy never pools."""
    with composite_backend.use_backend("native"):
        _pooled_case(pool, style, monkeypatch)


def _pooled_case(executor, style, monkeypatch):
    ran = {"chunks": 0, "patch": 0}
    real_chunks, real_patch = _ChunkCacheBase._pooled_chunks, _ChunkCacheBase._pooled_patch

    def chunks(self, *a, **k):
        ran["chunks"] += 1
        return real_chunks(self, *a, **k)

    def patch(self, *a, **k):
        ran["patch"] += 1
        return real_patch(self, *a, **k)

    monkeypatch.setattr(_ChunkCacheBase, "_pooled_chunks", chunks)
    monkeypatch.setattr(_ChunkCacheBase, "_pooled_patch", patch)
    scenario = _scenario()
    cache = _make(style, scenario)
    cache.set_chunk_executor(executor, WORKERS)
    cache.POOL_MIN_PATCH_TILE_AREA = 0
    for k in KS:
        _assert_same(_reduced(cache, k), _oracle(_make(style, scenario), k), f"{style} pooled build k={k}")
    assert ran["chunks"], "no pooled render_rect ran: the case proves nothing"
    assert _stale_after(style, "patch", cache=cache, scenario=scenario) == []
    assert ran["patch"], "no pooled patch ran: the case proves nothing"


@pytest.mark.parametrize("style", STYLES)
def test_reduced_hits_keep_their_parents_resident(env, style, monkeypatch):
    """With max_bytes holding the coarsest level plus half its neighbour, chunk
    traffic on the neighbour would age the coarsest level out if a reduced hit
    were not a use of its parents, and every read would rebuild."""
    scenario = _scenario()
    cache = _make(style, scenario)
    mip = _coarsest(cache)
    neighbour = cache.mip_levels()[1]
    _reduced(cache, 1)
    coarse_bytes = sum(chunk.nbytes for key, chunk in cache._cache.items() if key[0] == mip)
    w, h = cache.canvas_dims(neighbour)
    keys = [(neighbour, cx, cy) for cy in range(-(-h // cache.chunk_px)) for cx in range(-(-w // cache.chunk_px))]
    halves = keys[: len(keys) // 2], keys[len(keys) // 2 :]
    sizes = [sum(cache.get_chunk(*key).nbytes for key in half) for half in halves]
    cache.max_bytes = coarse_bytes + max(sizes)
    cache._evict()
    _reduced(cache, 1)  # rebuilds whatever the sizing pass evicted
    builds = _count_builds(monkeypatch)
    for i in range(4):
        for key in halves[i % 2]:
            cache.get_chunk(*key)
        _reduced(cache, 1)
    assert builds == [], f"{len(builds)} reduced rebuilds under neighbour traffic"


# --- injects: each must make a case above fail ---------------------------------


def test_inject_a_write_slice_without_its_stamp_is_caught(env, monkeypatch):
    def write_slice(self, chunk, mip, cx, cy, ix0, iy0, ix1, iy1, pixels):
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        chunk[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = pixels

    monkeypatch.setattr(_ChunkCacheBase, "_write_slice", write_slice)
    assert _stale_after("stepped", "patch") == [3, 2, 1]


def test_inject_validating_k_against_its_k_minus_1_entries_is_caught(env, monkeypatch):
    """Round 1's blocker: a k-1 entry is rebuilt only when read, so after an
    edit it is still present, with its old pixels, under a k entry."""
    real = _ChunkCacheBase._reduced_valid

    def by_sub_entries(self, entry, key):
        mip, k, cx, cy = key
        if k == 1:
            return real(self, entry, key)
        rw, rh = self.reduced_dims(mip, k - 1)
        size = self.chunk_px
        subs = [
            (mip, k - 1, 2 * cx + dx, 2 * cy + dy) for dy in (0, 1) for dx in (0, 1)
            if (2 * cx + dx) * size < rw and (2 * cy + dy) * size < rh
        ]
        return all(sub in self._reduced for sub in subs)

    monkeypatch.setattr(_ChunkCacheBase, "_reduced_valid", by_sub_entries)
    stale = _stale_after("stepped", "patch")
    assert 3 in stale and 2 in stale and 1 not in stale, stale


def test_inject_validating_by_stamp_alone_is_caught(env, monkeypatch):
    def stamp_alone(self, entry, key):
        stamps = self._chunk_stamp
        return entry.dims == self.canvas_dims(key[0]) and all(stamps.get(p) == s for p, s in entry.parents)

    monkeypatch.setattr(_ChunkCacheBase, "_reduced_valid", stamp_alone)
    assert _evicted_then_edited_stale("stepped") == [3, 2, 1]


# --- paint(): k, tiling, the bounding rect, alignment, the look ------------------


def _residual_scale(residual: float, coarsest: int = -2) -> float:
    """The device scale at which the coarsest level's residual is `residual`."""
    return residual / 2.0**-coarsest


@pytest.mark.parametrize(
    ("residual", "k"),
    [(1.0, 0), (0.75, 0), (0.5000001, 0), (0.5, 1), (0.3, 1), (0.2500001, 1), (0.25, 2), (0.13, 2),
     (0.125, 3), (0.07, 3), (0.0625, 3), (0.01, 3)],
)
def test_reduction_count_in_residuals(residual, k):
    from descape.viewer_canvas import MAX_REDUCTIONS, reduction_for

    assert MAX_REDUCTIONS == 3
    assert reduction_for(-2, _residual_scale(residual)) == k
    if residual > 0.5 / 2**MAX_REDUCTIONS:
        assert 0.5 < residual * 2**k <= 1.0


def test_reduction_count_is_zero_at_a_degenerate_scale():
    from descape.viewer_canvas import reduction_for

    assert reduction_for(-2, 0.0) == 0
    assert reduction_for(-2, -0.3) == 0


@pytest.mark.gui
@pytest.mark.parametrize("residual", [0.45, 0.3, 0.2, 0.09])
def test_reduced_residual_lands_in_the_band_under_flat_iso(residual):
    """Flat's squash-then-rotate: k is read off the max-axis scale, so the
    reduced max-axis residual is in (0.5, 1] there too."""
    from PyQt5.QtGui import QTransform

    from descape.map_view import MapView
    from descape.viewer_canvas import _max_axis_scale, reduction_for

    t = QTransform()
    t.scale(1.0, MapView.ISO_VERTICAL_SQUASH)
    t.rotate(MapView.ISO_ROTATION_DEGREES)
    s = _max_axis_scale(t)
    t2 = QTransform.fromScale(_residual_scale(residual) / s, _residual_scale(residual) / s) * t
    k = reduction_for(-2, _max_axis_scale(t2))
    assert k >= 1
    assert 0.5 < _max_axis_scale(t2) * 4 * 2**k <= 1.0


@pytest.fixture
def canvas_scene():
    """make(cache) -> (scene, MapCanvasItem) over it."""
    from PyQt5.QtWidgets import QGraphicsScene

    from descape.viewer_canvas import MapCanvasItem

    conftest.ensure_qapp()
    made = []

    def make(cache, item_cls=MapCanvasItem):
        scene = QGraphicsScene()
        item = item_cls(cache)
        scene.addItem(item)
        made.append((scene, item))
        return scene, item

    yield make
    for scene, item in made:
        scene.removeItem(item)


def _render(scene, rect, zoom: float, fill: int = 0, pre=None) -> np.ndarray:
    """One scene.render() of `rect` (scene units) into an RGB32 target `zoom`
    times larger (the backing store's format), filled with `fill` first.
    `pre`: (painter transform, image size) applied under the scene's own mapping."""
    from PyQt5.QtGui import QImage, QPainter

    tw, th = round(rect.width() * zoom), round(rect.height() * zoom)
    image = QImage(*(pre[1] if pre else (tw, th)), QImage.Format_RGB32)
    image.fill(fill)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    if pre:
        painter.setTransform(pre[0])
    scene.render(painter, QRectF_(0, 0, tw, th), rect)
    painter.end()
    return qt_capture.qimage_rgb888_to_array(image.convertToFormat(QImage.Format_RGB888))


def _flat_iso(rect, zoom: float, off: float = 0.0):
    """MapView's Flat isometric transform, scale(1, .5) rotate(-45), translated so
    the whole target lands inside the image `off` device px from its corner."""
    from PyQt5.QtGui import QTransform

    from descape.map_view import MapView

    turn = QTransform()
    turn.scale(1.0, MapView.ISO_VERTICAL_SQUASH)
    turn.rotate(MapView.ISO_ROTATION_DEGREES)
    box = turn.mapRect(QRectF_(0, 0, round(rect.width() * zoom), round(rect.height() * zoom)))
    pre = QTransform.fromTranslate(off - box.left() + 1, off - box.top() + 1)
    pre.scale(1.0, MapView.ISO_VERTICAL_SQUASH)
    pre.rotate(MapView.ISO_ROTATION_DEGREES)
    return pre, (math.ceil(box.width()) + 3, math.ceil(box.height()) + 3)


def _pixel_scene_coords(rect, zoom: float, shape, pre=None) -> tuple[np.ndarray, np.ndarray]:
    """Scene coordinates of every device pixel centre of a _render() image."""
    from PyQt5.QtGui import QTransform

    tw, th = round(rect.width() * zoom), round(rect.height() * zoom)
    full = QTransform.fromTranslate(-rect.left(), -rect.top()) * QTransform.fromScale(tw / rect.width(), th / rect.height())
    if pre:
        full = full * pre[0]
    inv, ok = full.inverted()
    assert ok
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]] + 0.5
    return inv.m11() * xs + inv.m21() * ys + inv.dx(), inv.m12() * xs + inv.m22() * ys + inv.dy()


def QRectF_(*a):
    from PyQt5.QtCore import QRectF

    return QRectF(*a)


TILING_BLOCK_PX = 64


# Each block's own clipped draw rounds its spans apart from one draw's (<= 2 measured on these cases).
TILING_MAX_DIFF = 8


def _tiled_vs_single(canvas_scene, monkeypatch, source, level, residual, k, iso) -> np.ndarray:
    """Per-pixel max channel diff between a TILING_BLOCK_PX tiled paint and one
    undivided draw. `level` is "coarsest" (zoom from `residual`) or 0 (zoom = residual)."""
    from test_rgb32_blit import _NoiseCache

    from descape.viewer_canvas import MapCanvasItem

    if source == "noise":
        cache = _NoiseCache()
        # The point overload's 1:1 level 0 target over a whole 3000 px probe is too big per xdist worker.
        probe = QRectF_(300, 300, 700, 700) if level == 0 else QRectF_(300, 300, 3000, 3000)
    else:
        cache = _make(source, _scenario())
        probe = QRectF_(0, 0, *cache.canvas_dims(0))
    scene, item = canvas_scene(cache)
    mip = cache.mip_levels()[0] if level == "coarsest" else level
    zoom = _residual_scale(residual, mip) if level == "coarsest" else residual
    pre = _flat_iso(probe, zoom) if iso else None
    monkeypatch.setattr(MapCanvasItem, "PAINT_BLOCK_PX", 100_000)
    single = _render(scene, probe, zoom, pre=pre)
    assert (item._last_mip, item._last_reduction) == (mip, k)
    monkeypatch.setattr(MapCanvasItem, "PAINT_BLOCK_PX", TILING_BLOCK_PX)
    tiled = _render(scene, probe, zoom, pre=pre)
    assert (item._last_mip, item._last_reduction) == (mip, k)
    assert single.any()
    return np.abs(tiled.astype(int) - single.astype(int)).max(axis=-1)


@pytest.mark.gui
@pytest.mark.parametrize(
    ("source", "level", "residual", "k", "iso"),
    [
        ("stepped", "coarsest", 0.45, 1, False),
        ("stepped", "coarsest", 0.3, 1, False),
        ("stepped", "coarsest", 0.2, 2, False),
        ("noise", "coarsest", 0.3, 1, True),
        ("stepped", "coarsest", 0.8, 0, False),
        ("stepped", "coarsest", 0.95, 0, False),
        ("noise", 0, 0.7, 0, False),
        ("noise", "coarsest", 0.8, 0, True),
    ],
    ids=["0.45", "0.3", "0.2", "flat-iso-0.3", "k0-0.8", "k0-0.95", "level0-point-0.7", "flat-iso-k0-0.8"],
)
def test_block_tiling_matches_a_single_draw_on_textured_content(
    env, canvas_scene, monkeypatch, source, level, residual, k, iso
):
    """test_mip_viewer's tiled-equals-single shape on textured content (and, under
    Flat's rotated transform, on _NoiseCache), where a misplaced, unpadded or
    unclipped block would show. Every pixel is barred, block edges included: each
    block draws only its own core (TASK-031.75), so a later block's pad no longer
    overwrites a neighbour with edge-clamped samples. The bar covers each block's
    own span rounding, which a tiled paint keeps."""
    diff = _tiled_vs_single(canvas_scene, monkeypatch, source, level, residual, k, iso)
    assert int(diff.max()) <= TILING_MAX_DIFF, f"tiled differs from one draw by {int(diff.max())}"


@pytest.mark.gui
def test_block_tiling_control_sees_the_overwrite_seam(env, canvas_scene, monkeypatch):
    """The same harness with each block's clip ignored must exceed the bar, so a
    fixture that renders flat (which hid this seam on test_mip_viewer's blank
    fixture) fails here instead of passing the case above."""
    from descape.viewer_canvas import _BlockDraw

    monkeypatch.setattr(_BlockDraw, "__call__", _BlockDraw._draw)
    diff = _tiled_vs_single(canvas_scene, monkeypatch, "stepped", "coarsest", 0.8, 0, False)
    assert int(diff.max()) > TILING_MAX_DIFF, f"unclipped tiling differs by only {int(diff.max())}"


@pytest.mark.gui
@pytest.mark.parametrize(
    ("residual", "k", "iso"), [(0.45, 1, False), (0.2, 2, False), (0.12, 3, False), (0.3, 1, True)],
    ids=["0.45", "0.2", "0.12", "flat-iso-0.3"],
)
def test_no_pixel_outside_the_bounding_rect_changes(canvas_scene, residual, k, iso):
    """A coarsest level of 8n + 1 px (1017) leaves the reduced canvas
    overhanging boundingRect by up to 7 level px at k = 3; the trimmed source
    keeps every pixel whose centre lies outside it untouched, under Flat's
    rotated transform too. Each case sweeps the placement over a device pixel,
    so some placement puts a centre inside the overhang."""
    from test_rgb32_blit import _NoiseCache

    cache = _NoiseCache(side=4068)
    assert cache.canvas_dims(-2) == (1017, 1017)
    scene, item = canvas_scene(cache)
    bound = item.boundingRect()
    zoom = _residual_scale(residual)
    magenta = np.array([255, 0, 255], dtype=np.uint8)
    for step in range(8):
        if iso:
            rect = QRectF_(-64.0, -64.0, bound.width() + 128.0, bound.height() + 128.0)
            pre = _flat_iso(rect, zoom, off=step / 8)
        else:
            far = 64.0 + step / 8 / zoom
            rect = QRectF_(-64.0, -64.0, bound.width() + 64.0 + far, bound.height() + 64.0 + far)
            pre = None
        img = _render(scene, rect, zoom, fill=0xFF00FF, pre=pre)
        assert item._last_reduction == k
        sx, sy = _pixel_scene_coords(rect, zoom, img.shape, pre)
        # A pixel is outside when its centre maps outside boundingRect (by 1/50 device px, for rounding).
        eps = 0.02 / zoom
        outside = (sx < -eps) | (sy < -eps) | (sx > bound.width() + eps) | (sy > bound.height() + eps)
        assert (img[~outside] != magenta).any()
        ys, xs = np.nonzero(outside & np.any(img != magenta, axis=-1))
        assert ys.size == 0, f"step {step}: {ys.size} px outside boundingRect painted, e.g. y={ys[:3]} x={xs[:3]}"


class _OneDrawItem:
    """Built lazily (QGraphicsItem needs PyQt5): draws a given RGB32 image over
    boundingRect at `scale` scene units per image pixel, in one drawImage(),
    trimmed to boundingRect the same way paint() is."""

    @staticmethod
    def make(bound, pixels: np.ndarray, scale: float):
        from PyQt5.QtCore import QRectF
        from PyQt5.QtGui import QImage
        from PyQt5.QtWidgets import QGraphicsItem

        class Item(QGraphicsItem):
            def boundingRect(self):
                return bound

            def paint(self, painter, option, widget=None):
                h, w = pixels.shape[:2]
                data = np.ascontiguousarray(pixels)
                qimg = QImage(data.data, w, h, 3 * w, QImage.Format_RGB888).convertToFormat(QImage.Format_RGB32)
                sw, sh = min(float(w), bound.width() / scale), min(float(h), bound.height() / scale)
                painter.drawImage(QRectF(0, 0, sw * scale, sh * scale), qimg, QRectF(0, 0, sw, sh))

        return Item()


@pytest.mark.gui
@pytest.mark.parametrize("style", STYLES)
def test_a_k1_render_is_one_draw_of_the_box_reduced_level(env, canvas_scene, style):
    """Pins the half-pixel alignment: reduced pixel i covers parent pixels 2i
    and 2i+1, drawn at twice the level's scale from the canvas origin."""
    from PyQt5.QtWidgets import QGraphicsScene

    scenario = _scenario()
    cache = _make(style, scenario)
    scene, item = canvas_scene(cache)
    mip = _coarsest(cache)
    bound = item.boundingRect()
    zoom = _residual_scale(0.45, mip)
    got = _render(scene, bound, zoom)
    assert item._last_reduction == 1
    ref_scene = QGraphicsScene()
    ref_item = _OneDrawItem.make(bound, _oracle(_make(style, scenario), 1), cache.mip_scale(mip) * 2)
    ref_scene.addItem(ref_item)
    want = _render(ref_scene, bound, zoom)
    ref_scene.removeItem(ref_item)
    _assert_same(got, want, f"{style} k=1 scene render")


@pytest.mark.gui
@pytest.mark.parametrize("residual", [0.3, 0.45])
def test_the_reduction_is_nearer_an_area_average_than_bilinear_alone(env, canvas_scene, monkeypatch, residual):
    """The look proxy: against level 0 area-averaged to the device size, the
    mean error is lower with the reduction than with bilinear skipping texels."""
    from PIL import Image

    from descape import viewer_canvas

    scenario = _scenario()
    cache = _make("stepped", scenario)
    scene, item = canvas_scene(cache)
    probe = QRectF_(1024, 1024, 2048, 1536)
    zoom = _residual_scale(residual, _coarsest(cache))
    reduced = _render(scene, probe, zoom)
    assert item._last_reduction == 1
    monkeypatch.setattr(viewer_canvas, "reduction_for", lambda coarsest, scale: 0)
    plain = _render(scene, probe, zoom)
    assert item._last_reduction == 0
    level0 = _make("stepped", scenario).render_rect(1024, 1024, 1024 + 2048, 1024 + 1536, mip=0)
    h, w = reduced.shape[:2]
    truth = np.asarray(Image.fromarray(level0).resize((w, h), Image.BOX), dtype=int)
    mae_reduced = float(np.abs(reduced.astype(int) - truth).mean())
    mae_plain = float(np.abs(plain.astype(int) - truth).mean())
    assert mae_reduced < mae_plain, (mae_reduced, mae_plain)
