"""tools/bench_draw_stroke.py's phased patch reads each resident level's own state."""

from __future__ import annotations

from test_sloped_mip import _cache, _scenario, _whole

from descape import iso_geometry
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
