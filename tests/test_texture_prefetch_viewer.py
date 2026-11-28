"""The viewer starts the terrain texture prefetch (asset_source) at the top of
every render, before the chunk cache exists, with the map's terrain ids plus
the farm overlay's, and not at all with Terrain Textures off. The prefetch
itself is pinned in tests/test_asset_source.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import asset_source
from testkit.viewer_family import set_viewer_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture(autouse=True)
def _no_install_from_the_environment(monkeypatch):
    """The no-install cases stay so with AOE2DE_INSTALL_PATH set, which outranks the config conftest hides."""
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_FARM = 50  # draws as terrain 7 under the farm overlay
_FARM_TERRAIN = 7
_OTHER_TERRAIN = 10
_LAYER_ONLY = 21  # a terrain no tile has as its base


@pytest.fixture
def recorded(monkeypatch):
    """(events, window): prefetch calls and IsoChunkCache constructions, in order."""
    from descape import asset_source
    from descape.render_cache import IsoChunkCache

    events: list = []
    monkeypatch.setattr(
        asset_source, "prefetch_terrain_textures", lambda ids, _tile_pxs, back=False, angled=False: events.append(("prefetch", set(ids)))
    )

    class _RecordingCache(IsoChunkCache):
        def __init__(self, *args, **kwargs) -> None:
            events.append(("cache",))
            super().__init__(*args, **kwargs)

    set_viewer_global(monkeypatch, "IsoChunkCache", _RecordingCache)
    window = conftest.stepped_window(UNITS_FIXTURE, show=False)
    yield events, window
    conftest.close_window(window)


def test_a_load_prefetches_the_maps_terrain_ids_before_building_the_cache(recorded):
    events, _window = recorded
    assert events[:2] == [("prefetch", {0}), ("cache",)], events


def test_a_re_render_adds_the_farm_overlays_terrain_only_while_that_layer_draws(recorded):
    """TASK-693: a viewport poll fired inside the toggle's processEvents() warms the levels, and its
    prefetch must already leave the farm terrain out. A 0 ms poll makes that fire every time."""
    events, window = recorded
    window.map_view._viewport_poll_timer.setInterval(0)
    window.scenario.map_manager.terrain[5].terrain_id = _OTHER_TERRAIN
    next(unit for units in window.scenario.unit_manager.units for unit in units).unit_const = _FARM
    events.clear()
    window._render_current(reset_view=False)
    # The render's own prefetch is the one right before the cache; a poll's can come first.
    built = events.index(("cache",))
    assert built > 0 and events[built - 1] == ("prefetch", {0, _OTHER_TERRAIN, _FARM_TERRAIN}), events
    events.clear()
    window._on_layer_toggled("farm_overlay", False)
    window._render_current(reset_view=False)
    built = events.index(("cache",))
    assert built > 0 and events[built - 1] == ("prefetch", {0, _OTHER_TERRAIN}), events
    assert all(event == ("prefetch", {0, _OTHER_TERRAIN}) for event in events if event[0] == "prefetch"), events


def test_a_sprites_toggle_drops_the_farm_terrain_before_a_poll_can_warm(recorded):
    """TASK-693: the farm overlay draws only with sprites on, and Show Sprites does not re-render."""
    events, window = recorded
    window.map_view._viewport_poll_timer.setInterval(0)
    next(unit for units in window.scenario.unit_manager.units for unit in units).unit_const = _FARM
    window._render_current(reset_view=False)
    assert _FARM_TERRAIN in window._texture_ids
    events.clear()
    window._on_sprites_toggled(False)
    prefetched = [ids for kind, *rest in events if kind == "prefetch" for ids in rest]
    assert prefetched and all(_FARM_TERRAIN not in ids for ids in prefetched), events


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


def test_each_style_prefetches_the_texture_variant_it_draws(recorded, monkeypatch):
    """TASK-691: Stepped, Sloped and flattened Stepped read the angled variant, top-down Flat the
    sharp one. A wrong flag only shows as a silent GUI-thread reload, so it is pinned here."""
    _events, window = recorded
    flags: list = []
    monkeypatch.setattr(
        asset_source, "prefetch_terrain_textures", lambda ids, tile_pxs, back=False, angled=False: flags.append(angled)
    )
    for label, iso, want in (("Sloped", False, True), ("Flat", False, False), ("Flat", True, True), ("Stepped", False, True)):
        flags.clear()
        window.terrain_style_combo.setCurrentText(label)
        window.iso_action.setChecked(iso)
        window._render_current(reset_view=False)
        assert flags and set(flags) == {want}, (label, iso, window._render_style, flags)


def test_prefetch_ids_include_layered_terrains_only_while_blending_is_effective(recorded, monkeypatch):
    from descape import asset_source, render_paint

    _events, window = recorded
    terrain = window.scenario.map_manager.terrain
    terrain[5].layer = _OTHER_TERRAIN
    terrain[9].layer = _LAYER_ONLY
    terrain[11].layer = _LAYER_ONLY
    # No install here, so no blendomatic: a layer does not draw and is not fetched.
    assert not render_paint.blend_effective(window._layers)
    assert sorted(window._texture_prefetch_ids()) == [0]

    # Read only by blend_effective() below: nothing renders while it is set.
    blendomatic = {"value": None}
    fake = lambda: blendomatic["value"]  # noqa: E731
    fake.cache_clear = lambda: None
    monkeypatch.setattr(asset_source, "get_blendomatic", fake)
    blendomatic["value"] = object()
    assert render_paint.blend_effective(window._layers)
    assert sorted(window._texture_prefetch_ids()) == [0, _OTHER_TERRAIN, _LAYER_ONLY]  # no -1, no duplicate

    blendomatic["value"] = None
    window._on_layer_toggled("terrain_blend", False)
    blendomatic["value"] = object()
    assert sorted(window._texture_prefetch_ids()) == [0]  # the Terrain Blending row is off
    blendomatic["value"] = None


@pytest.mark.parametrize("preload", [True, False])
def test_a_settled_load_queues_every_ladder_level_behind_while_zoom_levels_preload(monkeypatch, preload):
    """GH #326: the finest levels' 1920 px resizes start after the first paint, not at their first visit.
    TASK-691: every prefetch of this Stepped window, the level warm and the ladder included, asks for the
    angled variant (a sharp one would cancel the angled loads already queued)."""
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    calls: list = []
    flags: list = []

    def _record(ids, tile_pxs, back=False, angled=False):
        calls.append((list(tile_pxs), back))
        flags.append(angled)

    monkeypatch.setattr(asset_source, "prefetch_terrain_textures", _record)
    monkeypatch.setattr(settings, "get_preload_zoom_levels", lambda: preload)
    window = conftest.stepped_window(UNITS_FIXTURE, show=True)
    try:
        for _ in range(10):
            QApplication.processEvents()
        ladder = [window._cache.mip_tile_px(m) for m in window._cache.mip_levels()]
        backs = [pxs for pxs, back in calls if back]
        assert backs == ([ladder] if preload else []), calls
        assert flags and all(flags), list(zip(calls, flags, strict=True))
    finally:
        conftest.close_window(window)


def test_terrain_textures_off_prefetches_nothing(recorded):
    events, window = recorded
    window._on_layer_toggled("terrain_textures", False)
    events.clear()
    window._render_current(reset_view=False)
    assert ("cache",) in events and not [e for e in events if e[0] == "prefetch"], events
