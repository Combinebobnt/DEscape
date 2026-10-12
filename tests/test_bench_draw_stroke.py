"""tools/bench_draw_stroke.py's phased patch reads each resident level's own state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_sloped_mip import BLACK, DESERT_SAND, MAP_W, _cache, _scenario, _whole

from descape import iso_geometry, render
from tools import bench_draw_stroke


def test_sloped_phased_patch_reads_the_resident_levels_own_proj_and_bboxes(monkeypatch) -> None:
    """Only level -1 is resident, so the bystander scan must use its proj and
    never build level 0 outside the level_rebuild timer."""
    cache = _cache(_scenario())
    _whole(cache, -1)
    assert {key[0] for key in cache._cache} == {-1}
    assert cache._levels[0].gen != cache._source_gen, "level 0 already built, so the check is vacuous"
    seen: list[iso_geometry.IsoProjection] = []
    real = iso_geometry.tiles_in_screen_rect

    def spy(x0, y0, x1, y1, w, h, proj):
        seen.append(proj)
        return real(x0, y0, x1, y1, w, h, proj)

    monkeypatch.setattr(iso_geometry, "tiles_in_screen_rect", spy)
    bench_draw_stroke._patch_phased(cache, "sloped", (0, 0, *cache.canvas_dims(0)), set())
    assert seen and all(p == cache._levels[-1].proj for p in seen)
    assert cache._levels[0].gen != cache._source_gen, "level 0 built by the bench, outside any timer"


def test_phased_patch_repaints_a_terrain_edit_like_patch_does() -> None:
    """The phased patch re-reads the terrain mirror before compositing, as
    patch_sources() does: the native composite and the blend keys read the
    mirror, so without it the bench times a repaint of the old terrain."""
    scn = _scenario(units=False)
    cache = _cache(scn)
    _whole(cache, 0)
    tile = scn.map_manager.terrain[7 * MAP_W + 9]
    tile.terrain_id = BLACK if tile.terrain_id == DESERT_SAND else DESERT_SAND
    changed: set = set()
    bbox = render.dirty_screen_bbox_sloped(scn, [7 * MAP_W + 9], cache.elevations, cache.proj, elevation_changed=changed)
    bench_draw_stroke._patch_phased(cache, "sloped", bbox, changed)
    assert cache._terrain_ids[7 * MAP_W + 9] == tile.terrain_id
    assert np.array_equal(_whole(cache, 0), _whole(_cache(scn), 0))


@pytest.mark.parametrize("style", ["stepped", "sloped", "flat", "flat-iso"])
def test_a_paint_stroke_runs_in_every_style(monkeypatch, style) -> None:
    """One whole Draw stroke per style reports a line (no install here, so blending is off)."""
    monkeypatch.setattr(bench_draw_stroke, "load_map_and_units", lambda _path: _scenario(units=False))
    line, keyed = bench_draw_stroke._run_stroke(Path("fake.aoe2scenario"), style, 9, False, blend=False)
    assert "ms/step=" in line and keyed == 0


def test_blend_off_reaches_every_style_cache() -> None:
    """--blend off must switch the row off on the cache itself: a bench that
    left it on would compare blending against blending."""
    for style in ("stepped", "sloped", "flat", "flat-iso"):
        for blend in (True, False):
            cache = bench_draw_stroke._make_cache(style, _scenario(units=False), False, blend)[0]
            assert cache.layers.terrain_blend is blend, (style, blend)
