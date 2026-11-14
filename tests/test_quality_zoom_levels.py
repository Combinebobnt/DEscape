"""GH #241: each graphics quality builds zoom levels only up to its own finest
tile_px (render_paint.FINEST_TILE_PIXELS), in Stepped, Sloped and Flat alike, so
Potatest and Potato stay cheap when zoomed in. Default reaches the 192 px
level (TASK-601), Enhanced stops at 128 until TASK-441. Top-down Flat's
power-of-two ladder stops at the largest power of two within that finest, so
at Default it stops at 128."""

from __future__ import annotations

import pytest
from test_sloped_mip import _cache as _sloped_cache
from test_sloped_mip import _scenario

from descape import iso_geometry, render_iso, render_paint, settings
from descape.render_cache import FlatChunkCache, IsoChunkCache

QUALITIES = sorted(settings.GRAPHICS_QUALITY_LABELS)


def test_low_qualities_stop_below_the_ladders_top():
    assert [render_paint.FINEST_TILE_PIXELS[q] for q in QUALITIES] == [32, 64, 192, 128]
    assert [render_paint.NATIVE_TIER_MAX[q] for q in QUALITIES] == [96, 96, 192, 96]
    assert max(render_paint.FINEST_TILE_PIXELS.values()) == iso_geometry.MIP_MAX_TILE_PIXELS


def test_the_cap_never_removes_level_0():
    # MIP_MIN_TILE_PIXELS bounds the coarse end as before.
    assert iso_geometry.mip_tile_px_candidates(64, 32) == {0: 64, -1: 32, -2: 16}
    assert iso_geometry.mip_tile_px_candidates(16, 32) == {0: 16, 1: 32}


def _finest(style: str) -> int:
    scn = _scenario(units=False)
    if style == "flat":
        cache = FlatChunkCache(scn, render_paint.tile_pixels_for_map(scn.map_manager.map_width, scn.map_manager.map_height))
    elif style == "stepped":
        elevations, proj = render_iso.elevations_and_proj(scn)
        cache = IsoChunkCache(scn, elevations, proj, proj.tile_px)
    else:
        cache = _sloped_cache(scn)
    return cache.mip_tile_px(cache.mip_levels()[-1])


@pytest.mark.parametrize("quality", QUALITIES)
@pytest.mark.parametrize("style", ["stepped", "sloped", "flat"])
def test_every_view_stops_at_its_qualitys_finest_level(style, quality):
    settings.set_graphics_quality(quality)
    finest = render_paint.FINEST_TILE_PIXELS[quality]
    want = 1 << (finest.bit_length() - 1) if style == "flat" else finest
    assert _finest(style) == want
