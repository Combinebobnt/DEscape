"""GH #120: a farm hidden by Show Buildings also loses its Farm Terrain
Overlay drape, in every view mode.

The iso paths run UnitFilter.matches() before the farm override
(render._resolve_unit_sprite), but nothing pinned it. Flat has no farm path at
all (no SpriteLayer, so no farm_by_tile), so there a farm is a plain mark the
filter already drops; that is confirmed here rather than assumed.

The real FARM const (50) is used, not a synthetic one: the gate is a
building_consts() membership test, and FARM resolves no .sld but does carry a
foundation terrain, so the drape path runs with no install.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, render, unit_kind
from descape.render import tile_pixels_for_map
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache
from descape.unit_filter import UnitFilter
from testkit.fakes import FakeScenario, SyntheticTile

MAP_W = MAP_H = 16
BACKGROUND_TERRAIN = 1
FARM_CONST = 50
TEXTURE_SIZE = 256


@dataclass
class PlacedUnit:
    """SyntheticUnit carries no `rotation`, which the sprite resolver reads."""

    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    reference_id: int = 1


def _scenario(*, with_farm: bool) -> FakeScenario:
    tiles = [
        SyntheticTile(x=x, y=y, elevation=(x + y) % 3, terrain_id=BACKGROUND_TERRAIN)
        for y in range(MAP_H)
        for x in range(MAP_W)
    ]
    units_by_player = [[] for _ in range(9)]
    if with_farm:
        units_by_player[1] = [PlacedUnit(x=5.0, y=5.0, unit_const=FARM_CONST)]
    return FakeScenario(MAP_W, MAP_H, tiles, units_by_player)


@pytest.fixture
def textures(monkeypatch):
    """Solid colours, so a leftover drape would show as the farm terrain's own colour."""
    farm_terrain = render._terrain_overlay_for(FARM_CONST)
    assert farm_terrain is not None, "FARM lost its foundation terrain -- this file would prove nothing"
    colours = {BACKGROUND_TERRAIN: (10, 20, 30), farm_terrain: (200, 150, 40)}
    arrays = {tid: np.full((TEXTURE_SIZE, TEXTURE_SIZE, 3), c, dtype=np.uint8) for tid, c in colours.items()}
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", arrays.get)


def _iso_cache(scn, **kwargs) -> IsoChunkCache:
    elevations, proj = render.elevations_and_proj(scn)
    return IsoChunkCache(scn, elevations, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _sloped_cache(scn, **kwargs) -> SlopedChunkCache:
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return SlopedChunkCache(
        scn, elevations, corner_rise, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs
    )


def _flat_cache(scn, **kwargs) -> FlatChunkCache:
    return FlatChunkCache(scn, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _whole_canvas(cache) -> np.ndarray:
    w, h = cache.canvas_dims(0)
    return cache.render_rect(0, 0, w, h)


def _sprites(cache):
    if isinstance(cache, IsoChunkCache):
        return cache._level(0).sprites
    return cache.sprites


def test_farm_is_a_building_const() -> None:
    assert FARM_CONST in unit_kind.building_consts()


@pytest.mark.parametrize("builder", [_iso_cache, _sloped_cache])
def test_a_hidden_farm_loses_its_drape_in_stepped_and_sloped(builder, textures) -> None:
    shown = builder(_scenario(with_farm=True), sprites=True)
    shown_canvas = _whole_canvas(shown).copy()
    assert _sprites(shown).farm_by_tile, "the default filter built no drape -- test would be vacuous"

    hidden = builder(_scenario(with_farm=True), sprites=True, unit_filter=UnitFilter(show_buildings=False))
    hidden_canvas = _whole_canvas(hidden)
    assert not _sprites(hidden).farm_by_tile
    assert not _sprites(hidden).skip_ids

    empty = _whole_canvas(builder(_scenario(with_farm=False), sprites=True))
    assert not np.array_equal(shown_canvas, empty)
    assert np.array_equal(hidden_canvas, empty)


@pytest.mark.parametrize("builder", [_iso_cache, _sloped_cache])
def test_toggling_show_buildings_on_a_live_cache_drops_and_restores_the_drape(builder, textures) -> None:
    cache = builder(_scenario(with_farm=True), sprites=True)
    before = _whole_canvas(cache).copy()
    cache.set_unit_filter(UnitFilter(show_buildings=False))
    _whole_canvas(cache)
    assert not _sprites(cache).farm_by_tile
    cache.set_unit_filter(UnitFilter())
    assert np.array_equal(before, _whole_canvas(cache))


def test_flat_has_no_drape_and_a_hidden_farm_leaves_the_bare_map(textures) -> None:
    shown = _whole_canvas(_flat_cache(_scenario(with_farm=True), sprites=True))
    hidden = _whole_canvas(
        _flat_cache(_scenario(with_farm=True), sprites=True, unit_filter=UnitFilter(show_buildings=False))
    )
    empty = _whole_canvas(_flat_cache(_scenario(with_farm=False), sprites=True))
    assert not np.array_equal(shown, empty), "the farm drew nothing in Flat -- test would be vacuous"
    assert np.array_equal(hidden, empty)
