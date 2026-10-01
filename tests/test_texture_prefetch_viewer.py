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


def test_terrain_textures_off_prefetches_nothing(recorded):
    events, window = recorded
    window._on_layer_toggled("terrain_textures", False)
    events.clear()
    window._render_current(reset_view=False)
    assert ("cache",) in events and not [e for e in events if e[0] == "prefetch"], events
