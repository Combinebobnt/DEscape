"""TASK-031.61 A5: an edit past STEPPED_FULL_RERENDER_THRESHOLD refreshes the
Stepped/Sloped cache in place (refresh_wholesale()) instead of building a new
one, so the cache, its elevations array and the sprite memos survive.

Oracles are test_terrain_units_viewer's A2 set: sprites on, every oracle
level resident and re-rendered right before each step (GOTCHAS: an evicted
chunk redraws fresh and hides a stale one), then units_by_tile and every
level's pixels against fresh builds, failing as `<step>: ...`. Every window
here ends with edit_history.mark_saved() before close()."""

from __future__ import annotations

import pytest
from test_sprite_toggle_viewer import sprite_install  # noqa: F401 -- fixture, requested by name
from test_terrain_units_viewer import (
    FOREST_OAK,
    STROKE_TILES,
    _check_against_fresh,
    _draw,
    _fresh_cache,
    _over_cap_window,
    _render_levels,
    tree_sprites,  # noqa: F401 -- fixture, requested by name
)

from descape import render

import conftest
from test_unit_sprites import CONST

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

BEACH = 2
BLOCK = [(x, y) for y in range(60, 71) for x in range(60, 71)]  # 121 tiles, raised to level 2
THRESHOLD = 50
REFRESHED = "refreshed the full map in place instead of patching"


def _pin_threshold(monkeypatch, value: int = THRESHOLD) -> None:
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "STEPPED_FULL_RERENDER_THRESHOLD", value)


def _units_window(style: str, monkeypatch):
    """_over_cap_window() with 8 oaks (STROKE_TILES), 12 sprite units well
    away from BLOCK and 2 on it, then a Set elevation stroke raising BLOCK
    to level 2 (patched step by step, under the threshold)."""
    window = _over_cap_window(style, monkeypatch)
    _draw(window, FOREST_OAK, STROKE_TILES)
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [1]):
        for i in range(12):
            model.add(1, CONST, 20.5 + 3 * i, 90.5)
        model.add(1, CONST, 64.5, 64.5)
        model.add(1, CONST, 67.5, 62.5)
    window._on_tool_selected("set_level")
    window.elevation_level_spin.setValue(2)
    window.on_edit_stroke_start()
    for x, y in BLOCK:
        window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()
    assert window.scenario.map_manager.get_tile(65, 65).elevation == 2
    _pin_threshold(monkeypatch)
    return window


def _pick_points(window):
    from PyQt5.QtCore import QPointF

    points = []
    for y in range(56, 75):
        for x in range(56, 75):
            polygon = window.map_view._tile_polygon(x, y)
            if polygon is not None:
                center = polygon.boundingRect().center()
                points.append(QPointF(center.x(), center.y()))
    return points


def _check_picks(window, step: str, points) -> None:
    """MapView's pick at each point equals a fresh build's."""
    from descape import iso_geometry
    from descape.render_cache import SlopedChunkCache

    fresh = _fresh_cache(window)
    for p in points:
        if isinstance(fresh, SlopedChunkCache):
            want = fresh.pick_tile(int(p.x()), int(p.y()))
        else:
            want = iso_geometry.screen_to_tile(int(p.x()), int(p.y()), fresh.elevations, fresh.proj)
        assert window.map_view._pick_tile(p) == want, f"{step}: pick at ({p.x()}, {p.y()})"


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_over_threshold_elevation_undo_and_redo_refresh_the_cache_in_place(
    style, tree_sprites, monkeypatch  # noqa: F811
) -> None:
    """The cache and its elevations array survive, the next walk re-resolves
    only the units whose elevation key moved (the memo was reused, not just
    kept), and every oracle level and the pick match a fresh build."""
    window = _units_window(style, monkeypatch)
    try:
        cache = window._cache
        elevations = cache.elevations
        units = sum(len(u) for u in window.scenario.unit_manager.units)
        assert units >= 22
        resolves, walks = [], []
        real_resolve, real_walk = render._resolve_unit_sprite, render.sprite_draws_by_anchor_sliced
        monkeypatch.setattr(render, "_resolve_unit_sprite", lambda *a, **k: (resolves.append(1), real_resolve(*a, **k))[1])
        monkeypatch.setattr(render, "sprite_draws_by_anchor_sliced", lambda *a, **k: (walks.append(1), real_walk(*a, **k))[1])
        for step, act in (("undo", window.undo), ("redo", window.redo)):
            _render_levels(window)
            points = _pick_points(window)
            for p in points:
                window.map_view._pick_tile(p)  # pick planes warm, so a stale one would answer
            del resolves[:], walks[:]
            act()
            assert REFRESHED in window.status_log.toPlainText().splitlines()[-2], f"{step}: did not refresh"
            assert window._cache is cache and cache.elevations is elevations, f"{step}: rebuilt the cache"
            assert window._iso_elevations is elevations and window.map_view._iso_elevations is elevations, step
            _render_levels(window)
            # Each level walked again, re-resolving at most the 2 units on BLOCK (0 where a memo
            # entry predates the raise), not all of them.
            assert walks, f"{step}: no level walked its units again"
            assert len(resolves) <= 2 * len(walks) < units, f"{step}: {len(resolves)} resolves, {len(walks)} walks"
            _check_against_fresh(window, step)
            _check_picks(window, step, _pick_points(window))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", ["undo", "redo"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_refresh_check_names_the_step_whose_elevations_went_stale(
    style, broken, tree_sprites, monkeypatch  # noqa: F811
) -> None:
    """Control: on step `broken` only, refresh_wholesale() evicts and re-reads
    as usual but drops the gen bump (Stepped) or corner/unit rebuild (Sloped),
    reporting no elevation moved; the check must fail there."""
    window = _units_window(style, monkeypatch)
    try:
        cache = window._cache
        real = cache.refresh_wholesale
        current = {}

        def refresh(dirty_indices, flatten=False):
            if current.get("step") != broken:
                return real(dirty_indices, flatten)
            from descape import render_cache

            moved = render_cache._write_dirty_elevations(cache.scenario, cache.elevations, cache.proj, dirty_indices)
            cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
            return moved

        monkeypatch.setattr(cache, "refresh_wholesale", refresh)
        with pytest.raises(AssertionError, match=f"{broken}: "):
            for step, act in (("undo", window.undo), ("redo", window.redo)):
                _render_levels(window)
                current["step"] = step
                act()
                _check_against_fresh(window, step)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("trees", [False, True], ids=["terrain", "trees"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_large_fill_and_its_undo_and_redo_refresh_in_place(style, trees, tree_sprites, monkeypatch) -> None:  # noqa: F811
    """A Paint Can past the threshold, terrain only or Trees on (the A2b
    tile + unit case, whose unit step stays wholesale), then its undo and redo."""
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 100_000)
    window = _over_cap_window(style, monkeypatch)
    try:
        window.paint_trees_check.setChecked(False)
        region = [(x, y) for y in range(20, 32) for x in range(20, 32)]
        _draw(window, BEACH, region)
        _pin_threshold(monkeypatch)
        window._on_tool_selected("fill")
        window.paint_trees_check.setChecked(trees)
        cache = window._cache
        acts = {"fill": lambda: window.on_fill(20, 20, 0), "undo": window.undo, "redo": window.redo}
        window.terrain_panel.set_terrain(FOREST_OAK)
        for step, act in acts.items():
            _render_levels(window)
            refreshes = window.status_log.toPlainText().count(REFRESHED)
            act()
            # This step's own line: the log keeps the earlier steps' refreshes.
            assert window.status_log.toPlainText().count(REFRESHED) == refreshes + 1, f"{step}: did not refresh"
            assert window._cache is cache, f"{step}: rebuilt the cache"
            _check_against_fresh(window, step)
        assert bool(window.scenario.unit_manager.units[0]) is trees
    finally:
        window.edit_history.mark_saved()
        window.close()
