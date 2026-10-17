"""The viewer starts the terrain texture prefetch (asset_source) at the top of
every render, before the chunk cache exists, with the map's terrain ids plus
the farm overlay's, and not at all with Terrain Textures off. The prefetch
itself is pinned in tests/test_asset_source.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_FARM = 50  # draws as terrain 7 under the farm overlay
_FARM_TERRAIN = 7
_OTHER_TERRAIN = 10
_LAYER_ONLY = 21  # a terrain no tile has as its base


@pytest.fixture
def recorded(monkeypatch):
    """(events, window): prefetch calls and IsoChunkCache constructions, in order."""
    from descape import asset_source, viewer

    events: list = []
    monkeypatch.setattr(asset_source, "prefetch_terrain_textures", lambda ids: events.append(("prefetch", set(ids))))

    class _RecordingCache(viewer.IsoChunkCache):
        def __init__(self, *args, **kwargs) -> None:
            events.append(("cache",))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(viewer, "IsoChunkCache", _RecordingCache)
    window = conftest.stepped_window(UNITS_FIXTURE, show=False)
    yield events, window
    conftest.close_window(window)


def test_a_load_prefetches_the_maps_terrain_ids_before_building_the_cache(recorded):
    events, _window = recorded
    assert events[:2] == [("prefetch", {0}), ("cache",)], events


def test_a_re_render_adds_the_farm_overlays_terrain_only_while_that_layer_draws(recorded):
    events, window = recorded
    window.scenario.map_manager.terrain[5].terrain_id = _OTHER_TERRAIN
    next(unit for units in window.scenario.unit_manager.units for unit in units).unit_const = _FARM
    events.clear()
    window._render_current(reset_view=False)
    assert events[:2] == [("prefetch", {0, _OTHER_TERRAIN, _FARM_TERRAIN}), ("cache",)], events
    events.clear()
    window._on_layer_toggled("farm_overlay", False)
    window._render_current(reset_view=False)
    assert events[0] == ("prefetch", {0, _OTHER_TERRAIN}), events


def test_top_down_flat_prefetches_the_farm_overlays_terrain_too(recorded):
    """GH #150: top-down Flat drapes farms, so it fetches their terrain."""
    _events, window = recorded
    next(unit for units in window.scenario.unit_manager.units for unit in units).unit_const = _FARM
    window.terrain_style_combo.setCurrentText("Flat")
    window.iso_action.setChecked(False)
    assert window._render_style == "flat", "fixture no longer reaches real Flat"
    assert window._sprites_enabled and window._layers.farm_overlay
    assert _FARM_TERRAIN in window._texture_prefetch_ids()
    window._on_layer_toggled("farm_overlay", False)
    assert _FARM_TERRAIN not in window._texture_prefetch_ids()


def test_prefetch_ids_include_layered_terrains_only_while_blending_is_effective(recorded, monkeypatch):
    from descape import asset_source, render

    _events, window = recorded
    terrain = window.scenario.map_manager.terrain
    terrain[5].layer = _OTHER_TERRAIN
    terrain[9].layer = _LAYER_ONLY
    terrain[11].layer = _LAYER_ONLY
    # No install here, so no blendomatic: a layer does not draw and is not fetched.
    assert not render.blend_effective(window._layers)
    assert sorted(window._texture_prefetch_ids()) == [0]

    # Read only by blend_effective() below: nothing renders while it is set.
    blendomatic = {"value": None}
    fake = lambda: blendomatic["value"]  # noqa: E731
    fake.cache_clear = lambda: None
    monkeypatch.setattr(asset_source, "get_blendomatic", fake)
    blendomatic["value"] = object()
    assert render.blend_effective(window._layers)
    assert sorted(window._texture_prefetch_ids()) == [0, _OTHER_TERRAIN, _LAYER_ONLY]  # no -1, no duplicate

    blendomatic["value"] = None
    window._on_layer_toggled("terrain_blend", False)
    blendomatic["value"] = object()
    assert sorted(window._texture_prefetch_ids()) == [0]  # the Terrain Blending row is off
    blendomatic["value"] = None


def test_terrain_textures_off_prefetches_nothing(recorded):
    events, window = recorded
    window._on_layer_toggled("terrain_textures", False)
    events.clear()
    window._render_current(reset_view=False)
    assert ("cache",) in events and not [e for e in events if e[0] == "prefetch"], events
