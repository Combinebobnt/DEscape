"""A whole-canvas invalidate for a change that is not terrain (unit filter,
layers, grid, sprites, a unit mutation, a player recolour) skips the
terrain-id mirror re-read, which costs 2.9 ms at 240x240 and 11 ms at 480x480
on the native backend. Patched at class level, so the guard holds on either
backend (the mirror itself only exists on native)."""

from __future__ import annotations

import pytest

from descape import render_cache
from descape.grid_overlay import grid_bake
from descape.render import elevations_and_proj, sloped_elevations_and_proj, tile_pixels_for_map
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.unit_filter import UnitFilter
from descape.view_layers import LayerState

import conftest


@pytest.fixture
def reads(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(
        render_cache._ChunkCacheBase, "_refresh_terrain_ids", lambda self, bbox: calls.append((type(self).__name__, bbox))
    )
    return calls


def _cache(style: str):
    scenario = load_map_and_units(BLANK_TEMPLATE_PATH)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = elevations_and_proj(scenario)
        return IsoChunkCache(scenario, elevations, proj, tile_px, sprites=False)
    if style == "sloped":
        elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
        return SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=False)
    return FlatChunkCache(scenario, tile_px, sprites=False)


# Each flips a field the cache compares by value, so none of them no-ops.
_NON_TERRAIN_SETTERS = {
    "set_unit_filter": lambda cache: cache.set_unit_filter(UnitFilter(show_trees=False)),
    "set_layers (paint-time)": lambda cache: cache.set_layers(LayerState(terrain_textures=False)),
    "set_layers (build-time)": lambda cache: cache.set_layers(LayerState(small_trees=True)),
    "set_grid": lambda cache: cache.set_grid(grid_bake(True, 60, 2)),
    "set_sprites_enabled": lambda cache: cache.set_sprites_enabled(True),
}


@pytest.mark.parametrize("style", ["stepped", "sloped", "flat"])
@pytest.mark.parametrize("setter", list(_NON_TERRAIN_SETTERS))
def test_a_non_terrain_whole_canvas_invalidate_skips_the_mirror_reread(reads, style, setter) -> None:
    cache = _cache(style)
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
    assert reads, "the default invalidate did not reach the recorder -- vacuous"
    reads.clear()
    epoch = cache._mutation_epoch
    _NON_TERRAIN_SETTERS[setter](cache)
    assert cache._mutation_epoch > epoch, f"{setter} evicted nothing -- vacuous"
    assert reads == [], f"{style} {setter} re-read the terrain-id mirror"


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_the_viewer_unit_and_recolour_invalidates_skip_the_mirror_reread(reads, monkeypatch) -> None:
    from descape import viewer

    window = conftest.blank_window()
    try:
        assert window._cache is not None
        reads.clear()
        window._after_unit_mutation()
        assert reads == [], "_after_unit_mutation's wholesale invalidate re-read the mirror"

        window._ensure_option_edits()
        monkeypatch.setattr(viewer, "refresh_player_render_context", lambda *_args: True)
        epoch = window._cache._mutation_epoch
        window._apply_player_color_change()
        assert window._cache._mutation_epoch > epoch, "the recolour evicted nothing -- vacuous"
        assert reads == [], "_apply_player_color_change re-read the mirror"
    finally:
        conftest.close_window(window)
