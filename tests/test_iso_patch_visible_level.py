"""TASK-031.73: a Stepped or Sloped terrain patch composites the level in view
only. patch(levels=...) and patch_composite(levels=...) drop every other
resident level's chunks under the bbox, and the viewer passes the level in
view (plus a placeholder's source level, which test_placeholder_paint.py's
T3e pins), so no level is left stale and a stroke step costs one level.

The oracle is equality with a fresh cache built on the edited scenario, read
through render_rect() (so _level() flushes any deferred work first)."""

from __future__ import annotations

import numpy as np
import pytest
from test_farm_terrain import fake_farm  # noqa: F401 -- a fixture
from test_native_composite import (
    EDIT_X,
    EDIT_Y,
    NEW_TERRAIN_ID,
    _edit_cache,
    native_kernel,  # noqa: F401 -- a fixture
    synthetic_textures,  # noqa: F401 -- a fixture
)
from test_pooled_render_rect import _scenario_for
from test_sprite_edit_bbox import sprite_install  # noqa: F401

from descape import composite_backend, iso_geometry, render_bbox, render_iso, render_paint, render_sloped

import conftest

LEVELS = (-1, 0, 1)
BRUSH = [(EDIT_X + dx, EDIT_Y + dy) for dx in range(-2, 3) for dy in range(-2, 3)]
WARM_PAD = 1200  # REFERENCE px around the edit warmed at each level: chunks inside and outside the bbox


@pytest.fixture(params=["native", "numpy"])
def backend(request):
    if request.param == "native":
        request.getfixturevalue("native_kernel")
    with composite_backend.use_backend(request.param):
        yield request.param


def _dirty_bbox(cache, style: str, scenario, tiles, changed: set) -> tuple[int, int, int, int]:
    """The viewer's funnel: syncs cache.elevations, fills `changed`."""
    w = scenario.map_manager.map_width
    fn = render_bbox.dirty_screen_bbox_iso if style == "stepped" else render_bbox.dirty_screen_bbox_sloped
    bbox = fn(
        scenario, [y * w + x for x, y in tiles], cache.elevations, cache.proj,
        with_units=True, with_sprites=cache.sprites_enabled, elevation_changed=changed,
    )
    assert bbox is not None
    return bbox


def _region(scenario, cache) -> tuple[int, int, int, int]:
    """A REFERENCE rect around the edit, wider than its bbox."""
    bbox = _dirty_bbox(cache, cache.style, scenario, BRUSH, set())
    cw, ch = cache.canvas_dims(0)
    return (
        max(bbox[0] - WARM_PAD, 0), max(bbox[1] - WARM_PAD, 0), min(bbox[2] + WARM_PAD, cw), min(bbox[3] + WARM_PAD, ch)
    )


def _read(cache, mip: int, region) -> np.ndarray:
    return cache.render_rect(*cache._bbox_to_level(mip, region), mip=mip)


def _warm(cache, region) -> None:
    assert set(LEVELS) <= set(cache.mip_levels()), f"ladder {cache.mip_levels()} lacks {LEVELS}"
    cache.max_bytes *= 8  # every level stays resident
    for mip in LEVELS:
        _read(cache, mip, region)


def _fresh(style: str, scenario, sprites: bool):
    return _edit_cache(style, scenario, sprites=sprites)


def _keys_under(cache, mip: int, bbox) -> set:
    rng = cache.chunk_range_under(bbox, mip)
    assert rng is not None
    cx0, cy0, cx1, cy1 = rng
    return {(mip, cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)}


def _paint(scenario, tiles) -> None:
    mm = scenario.map_manager
    for x, y in tiles:
        tile = mm.terrain[y * mm.map_width + x]
        tile.terrain_id, tile.layer = NEW_TERRAIN_ID, -1


def _raise(scenario, tiles) -> None:
    mm = scenario.map_manager
    for x, y in tiles:
        tile = mm.terrain[y * mm.map_width + x]
        tile.elevation = min(tile.elevation + 1, iso_geometry.MAX_ELEVATION)


def _assert_cached_like_fresh(cache, fresh) -> None:
    """Every chunk the patch left cached equals a fresh cache's: a stale one
    at a level not in view fails here, naming its level."""
    for key, chunk in list(cache._cache.items()):
        want = fresh.get_chunk(*key)
        if not np.array_equal(chunk, want):
            pytest.fail(f"mip {key[0]} chunk {key[1:]} differs from a fresh cache: a stale chunk was kept")


def _assert_evicted_others(cache, before: dict, bbox, visible=(0,)) -> None:
    kept_outside = 0
    for key, chunk in before.items():
        mip = key[0]
        if mip in visible:
            assert cache._cache.get(key) is chunk, f"{key}: a visible-level chunk was replaced or evicted"
        elif key in _keys_under(cache, mip, bbox):
            assert key not in cache._cache, f"{key}: a chunk under the bbox at a level not in view survived"
        else:
            assert cache._cache.get(key) is chunk, f"{key}: a chunk outside the bbox was touched"
            kept_outside += 1
    assert kept_outside, "no chunk outside the bbox at another level -- vacuous"
    assert any(key[0] not in visible and key in _keys_under(cache, key[0], bbox) for key in before), "vacuous"
    assert cache._cache_bytes == sum(chunk.nbytes for chunk in cache._cache.values())


def _assert_like_fresh(cache, fresh, region) -> None:
    for mip in LEVELS:
        got, want = _read(cache, mip, region), _read(fresh, mip, region)
        assert want.any(), f"mip {mip} rendered black -- vacuous"
        if not np.array_equal(got, want):
            ys, _xs = np.nonzero(np.any(got != want, axis=-1))
            pytest.fail(f"mip {mip}: {ys.size} px differ from a fresh cache")


@pytest.mark.parametrize("style", ["stepped", "sloped"])
@pytest.mark.parametrize("edit", ["terrain", "elevation"])
def test_a_patch_given_levels_evicts_the_others_under_the_bbox(
    style, edit, backend, sprite_install, fake_farm, synthetic_textures,  # noqa: F811
) -> None:
    """T1 (terrain only) and T2 (elevation changing: Stepped's pending path,
    and Sloped): mip 0 patched in place, -1 and 1 drop their chunks under the
    bbox and keep the rest, and every level reads back equal to a fresh
    cache. On Sloped a terrain-only step leaves every pick plane alone."""
    scenario = _scenario_for(style)
    cache = _edit_cache(style, scenario, sprites=False)
    region = _region(scenario, cache)
    _warm(cache, region)
    planes = None
    if style == "sloped":
        cx, cy = (region[0] + region[2]) // 2 // cache.chunk_px, (region[1] + region[3]) // 2 // cache.chunk_px
        cache._pick_plane(cx, cy)
        planes = dict(cache._pick_planes)
        assert planes, "no pick plane built -- vacuous"
    before = dict(cache._cache)

    (_paint if edit == "terrain" else _raise)(scenario, BRUSH)
    changed: set = set()
    bbox = _dirty_bbox(cache, style, scenario, BRUSH, changed)
    assert bool(changed) is (edit == "elevation")
    cache.patch(bbox, elevation_changed=changed, levels=(0,), rebuild_levels=(0,))
    fresh = _fresh(style, scenario, False)
    _assert_cached_like_fresh(cache, fresh)

    if edit == "terrain":
        _assert_evicted_others(cache, before, bbox)
    else:
        for key in before:
            if key[0] != 0 and key in _keys_under(cache, key[0], bbox):
                assert key not in cache._cache, f"{key}: survived an elevation edit at a level not in view"
        assert cache._cache_bytes == sum(chunk.nbytes for chunk in cache._cache.values())
    if planes is not None and edit == "terrain":
        assert all(cache._pick_planes.get(key) is plane for key, plane in planes.items()), "a pick plane moved"
    _assert_like_fresh(cache, fresh, region)


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_patch_composite_given_levels_evicts_the_others(
    style, backend, sprite_install, fake_farm, synthetic_textures,  # noqa: F811
) -> None:
    """T5's cache half: patch_sources() then patch_composite(levels=) (the A4
    deferred composite's shape) evicts the others under the bbox."""
    scenario = _scenario_for(style)
    cache = _edit_cache(style, scenario, sprites=False)
    region = _region(scenario, cache)
    _warm(cache, region)
    before = dict(cache._cache)
    _paint(scenario, BRUSH)
    bbox = _dirty_bbox(cache, style, scenario, BRUSH, set())
    cache.patch_sources(bbox, elevation_changed=set())
    cache.patch_composite(bbox, levels=(0,), rebuild_levels=(0,))
    _assert_cached_like_fresh(cache, _fresh(style, scenario, False))
    _assert_evicted_others(cache, before, bbox)
    _assert_like_fresh(cache, _fresh(style, scenario, False), region)


# --- Viewer (T3, T4, T5) ----------------------------------------------------------------------


def _iso_window(style: str, sprites: bool):
    from PyQt5.QtGui import QTransform

    window = conftest.shown_terrain_window()
    window.show_sprites_action.setChecked(sprites)
    window.terrain_style_combo.setCurrentText("Sloped")
    if style == "Stepped":
        window.terrain_style_combo.setCurrentText("Stepped")
    _hold_warms(window)
    view = window.map_view
    view.setTransform(QTransform.fromScale(1, 1))
    view.centerOn(view.sceneRect().center())
    _pump()
    _hold_warms(window)
    cache = window._cache
    assert cache.style == style.lower() and cache.sprites_enabled is sprites
    assert view.viewport_chunk_target()[0] == 0, "not at 1:1"
    cache.max_bytes *= 8
    return window


def _pump() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()


def _hold_warms(window) -> None:
    window._placeholder_pending_timer.stop()
    window.map_view.stop_viewport_poll()
    window._cancel_warms()


def _view_region(window) -> tuple[int, int, int, int]:
    view = window.map_view
    rect = view.mapToScene(view.viewport().rect()).boundingRect()
    cw, ch = window._cache.canvas_dims(0)
    return max(int(rect.left()), 0), max(int(rect.top()), 0), min(int(rect.right()), cw), min(int(rect.bottom()), ch)


def _view_tiles(window, n: int = 4) -> list[tuple[int, int]]:
    view = window.map_view
    vp = view.viewport().rect()
    tiles = []
    for i in range(n):
        tile = view._pick_tile(view.mapToScene(vp.width() // 2 + 12 * i, vp.height() // 2))
        assert tile is not None
        tiles.append(tile)
    return tiles


def _stroke(window, tiles) -> None:
    mm = window.scenario.map_manager
    window._on_tool_selected("draw")
    window.terrain_panel.set_terrain(2 if mm.get_tile(*tiles[0]).terrain_id != 2 else 15)
    window.brush_size_spin.setValue(3)
    window.on_edit_stroke_start()
    for tile in tiles:
        window.on_edit_stroke_tiles([tile], 0)
    window.on_edit_stroke_end()


def _patch_spy(window, monkeypatch) -> list:
    cache = window._cache
    seen = []
    real = cache.patch

    def spy(bbox, elevation_changed=None, levels=None, rebuild_levels=None):
        seen.append((bbox, levels))
        return real(bbox, elevation_changed, levels, rebuild_levels=rebuild_levels)

    monkeypatch.setattr(cache, "patch", spy)
    return seen


def _window_fresh(window):
    """A fresh chunk cache of the window's style on its edited scenario."""
    from descape.render_cache import IsoChunkCache, SlopedChunkCache

    scenario = window.scenario
    mm = scenario.map_manager
    tile_px = render_paint.tile_pixels_for_map(mm.map_width, mm.map_height)
    cache = window._cache
    kwargs = {"sprites": cache.sprites_enabled, "layers": cache.layers}
    if cache.style == "sloped":
        elevations, corner_rise, proj = render_sloped.sloped_elevations_and_proj(scenario)
        fresh = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, **kwargs)
    else:
        elevations, proj = render_iso.elevations_and_proj(scenario)
        fresh = IsoChunkCache(scenario, elevations, proj, tile_px, **kwargs)
    fresh.set_grid(cache.grid)
    return fresh


def _warm_window(window, region) -> None:
    cache = window._cache
    assert set(LEVELS) <= set(cache.mip_levels()), cache.mip_levels()
    for mip in LEVELS:
        _read(cache, mip, region)


@pytest.mark.gui
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_draw_stroke_patches_the_level_in_view_only(style, sprites, monkeypatch) -> None:
    """T3: every step's patch names the level in view, mip 0's chunks under
    the stroke stay the same objects, the other levels' are gone, and every
    level then reads back equal to a fresh cache."""
    window = _iso_window(style, sprites)
    try:
        cache = window._cache
        region = _view_region(window)
        _warm_window(window, region)
        tiles = _view_tiles(window)
        before = dict(cache._cache)
        seen = _patch_spy(window, monkeypatch)
        _stroke(window, tiles)
        assert seen, "no patch() ran -- vacuous"
        fresh = _window_fresh(window)
        _assert_cached_like_fresh(cache, fresh)
        assert all(levels == (0,) for _bbox, levels in seen), [levels for _b, levels in seen]
        union = (
            min(b[0] for b, _ in seen), min(b[1] for b, _ in seen), max(b[2] for b, _ in seen), max(b[3] for b, _ in seen)
        )
        under = {key for key in before if key in _keys_under(cache, key[0], union)}
        assert any(key[0] == 0 for key in under) and any(key[0] != 0 for key in under), "vacuous"
        for key in under:
            if key[0] == 0:
                assert cache._cache.get(key) is before[key], f"{key}: a visible chunk was replaced"
            else:
                assert key not in cache._cache, f"{key}: survived under the stroke at a level not in view"
        _assert_like_fresh(cache, fresh, region)
    finally:
        _hold_warms(window)
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_with_no_viewport_target_every_level_patches_in_place(style, monkeypatch) -> None:
    """T4: viewport_chunk_target pinned to None, the patch gets levels=None
    and every level's chunks under the stroke stay the same objects."""
    window = _iso_window(style, False)
    try:
        cache = window._cache
        region = _view_region(window)
        _warm_window(window, region)
        tiles = _view_tiles(window)
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: None)
        before = dict(cache._cache)
        seen = _patch_spy(window, monkeypatch)
        _stroke(window, tiles)
        assert seen and all(levels is None for _bbox, levels in seen)
        assert {mip for mip, _cx, _cy in before} == set(LEVELS)
        assert all(cache._cache.get(key) is chunk for key, chunk in before.items()), "a chunk was evicted"
        _assert_like_fresh(cache, _window_fresh(window), region)
    finally:
        _hold_warms(window)
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_deferred_terrain_composite_patches_the_level_in_view_only(style) -> None:
    """T5: _composite_deferred_terrain(bbox) after a terrain mutation and
    patch_sources(), three levels warm: the others evict under the bbox, and
    every level reads back equal to a fresh cache."""
    window = _iso_window(style, False)
    try:
        cache = window._cache
        region = _view_region(window)
        _warm_window(window, region)
        tiles = _view_tiles(window, 2)
        scenario = window.scenario
        _paint(scenario, tiles)
        changed: set = set()
        fn = render_bbox.dirty_screen_bbox_iso if style == "Stepped" else render_bbox.dirty_screen_bbox_sloped
        w = scenario.map_manager.map_width
        bbox = fn(
            scenario, [y * w + x for x, y in tiles], window._iso_elevations, window._iso_proj,
            with_units=True, with_sprites=False, elevation_changed=changed,
        )
        assert bbox is not None and not changed
        before = dict(cache._cache)
        cache.patch_sources(bbox, elevation_changed=set())
        window._composite_deferred_terrain(bbox)
        _assert_cached_like_fresh(cache, _window_fresh(window))
        _assert_evicted_others(cache, before, bbox)
        _assert_like_fresh(cache, _window_fresh(window), region)
    finally:
        _hold_warms(window)
        conftest.close_window(window)
