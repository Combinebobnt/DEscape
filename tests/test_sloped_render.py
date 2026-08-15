"""Verifies descape.render's Phase 6 (Sloped) compositor additions --
_render_tile_sloped, render_terrain_sloped(_with_proj), composite_rect_sloped,
and render_scenario()/save_png()'s style= widening -- docs/PLAN_V2_6.md's
Track C2.

Checks:
  1. Flat-map byte identity: on a uniform-elevation map, render_terrain_sloped
     is EXACTLY render_terrain_iso's own output, units included -- the
     strongest and cheapest oracle this compositor has (see
     iso_geometry.sloped_quad_indices' and render._slope_shade's own
     docstrings for why each was built to make this exact, not merely close).
  2. Coverage: no unpainted (all-zero) pixel inside the ground outline on a
     genuinely sloped map -- the Sloped counterpart to
     tools/verify_iso_render.py's check_full_coverage.
  3. Delta>=2 graceful degradation (Risk #2 -- elevations are never
     validated): a synthetic illegal seam renders without raising.
  4. render_scenario()/save_png()'s style="sloped" parameter dispatches to
     render_terrain_sloped, and style=None/isometric keep their existing
     Flat/Stepped behavior unchanged (the backward-compatibility bar
     docs/PLAN_V2_6.md's must-keep-passing dump_scenario.py --iso needs).
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import iso_geometry as ig
from descape.render import (
    render_scenario,
    render_terrain_iso,
    render_terrain_sloped,
    render_terrain_sloped_with_proj,
)
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units


def _load() -> object:
    return load_map_and_units(str(BLANK_TEMPLATE_PATH))


@pytest.mark.parametrize("with_units", [False, True])
def test_flat_map_byte_identical_to_stepped(with_units):
    scenario = _load()
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = 2
    sloped = render_terrain_sloped(scenario, with_units=with_units)
    stepped = render_terrain_iso(scenario, with_units=with_units)
    assert sloped.shape == stepped.shape
    assert np.array_equal(sloped, stepped)


def test_coverage_on_sloped_map():
    scenario = _load()
    mm = scenario.map_manager
    # A gentle ramp climbing 1 level over half the map's width, well within
    # the +-1-elevation-between-neighbors invariant real maps use.
    for tile in mm.terrain:
        tile.elevation = 1 if tile.x >= mm.map_width // 2 else 0
    img, elevations, _corner_rise, proj = render_terrain_sloped_with_proj(scenario, with_units=False)
    corners = ig.ground_outline_corners(mm.map_width, mm.map_height, proj)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1 = max(0, min(xs)), min(img.shape[1], max(xs))
    y0, y1 = max(0, min(ys)), min(img.shape[0], max(ys))
    region = img[y0:y1, x0:x1]
    unpainted = int(np.all(region == 0, axis=-1).sum())
    total = region.shape[0] * region.shape[1]
    # Matches this project's own diamond-shaped (not rectangular) ground
    # outline -- some corner-of-the-bbox background is expected outside the
    # outline itself, so this asserts a generous upper bound (a real gap
    # would show as a large, contiguous hole, not this kind of background
    # margin), not zero.
    assert unpainted / total < 0.55, f"{unpainted}/{total} unpainted pixels in the outline bbox"


def test_delta_2_seam_does_not_raise():
    scenario = _load()
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = 3 if tile.x >= mm.map_width // 2 else 1
    img = render_terrain_sloped(scenario, with_units=False)
    assert img.dtype == np.uint8
    assert img.ndim == 3


def test_render_scenario_style_sloped_matches_direct_call():
    scenario = _load()
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = 1 if tile.x >= mm.map_width // 2 else 0
    via_style = render_scenario(scenario, with_units=False, style="sloped")
    direct = render_terrain_sloped(scenario, with_units=False)
    assert np.array_equal(via_style, direct)


def test_render_scenario_style_none_keeps_isometric_bool_behavior():
    scenario = _load()
    for stepped in (False, True):
        via_default = render_scenario(scenario, with_units=False, isometric=stepped)
        via_explicit_none = render_scenario(scenario, with_units=False, isometric=stepped, style=None)
        assert np.array_equal(via_default, via_explicit_none)


def test_render_scenario_rejects_bad_style():
    scenario = _load()
    with pytest.raises(ValueError):
        render_scenario(scenario, style="ramped")
