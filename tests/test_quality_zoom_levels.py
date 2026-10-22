"""GH #241: each graphics quality builds zoom levels only up to its own finest
tile_px (render.FINEST_TILE_PIXELS), in Stepped, Sloped and Flat alike, so
Potatest and Potato stay cheap when zoomed in. Default and Enhanced keep the
ladder's 128 px top."""

from __future__ import annotations

import pytest
from test_sloped_mip import _cache as _sloped_cache
from test_sloped_mip import _scenario

from descape import iso_geometry, render, settings
from descape.render_cache import FlatChunkCache, IsoChunkCache

QUALITIES = sorted(settings.GRAPHICS_QUALITY_LABELS)


def test_low_qualities_stop_below_the_ladders_top():
    assert [render.FINEST_TILE_PIXELS[q] for q in QUALITIES] == [32, 64, 128, 128]
    assert max(render.FINEST_TILE_PIXELS.values()) == iso_geometry.MIP_MAX_TILE_PIXELS


def test_the_cap_never_removes_level_0():
    # MIP_MIN_TILE_PIXELS bounds the coarse end as before.
    assert iso_geometry.mip_tile_px_candidates(64, 32) == {0: 64, -1: 32, -2: 16}
    assert iso_geometry.mip_tile_px_candidates(16, 32) == {0: 16, 1: 32}


def _finest(style: str) -> int:
    scn = _scenario(units=False)
    if style == "flat":
        cache = FlatChunkCache(scn, render.tile_pixels_for_map(scn.map_manager.map_width, scn.map_manager.map_height))
    elif style == "stepped":
        elevations, proj = render.elevations_and_proj(scn)
        cache = IsoChunkCache(scn, elevations, proj, proj.tile_px)
    else:
        cache = _sloped_cache(scn)
    return cache.mip_tile_px(cache.mip_levels()[-1])


@pytest.mark.parametrize("quality", QUALITIES)
@pytest.mark.parametrize("style", ["stepped", "sloped", "flat"])
def test_every_view_stops_at_its_qualitys_finest_level(style, quality):
    settings.set_graphics_quality(quality)
    assert _finest(style) == render.FINEST_TILE_PIXELS[quality]
