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
  5. _slope_shade alignment: the compositor gathers shading through
     sloped_quad_indices' uv_idx rather than slicing it to length. Since
     the resample made columns variable-length, slicing is a SILENT bug on
     a fixture where the two lengths happen to coincide -- see that test's
     own docstring, and tests/test_sloped_geometry.py's structural half.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from descape import iso_geometry as ig
from descape import render as render_mod
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
    #
    # DON'T "fix" the 0.55 threshold when the seam resample lands. It was
    # flagged as possibly failing for the right reason and measured not to:
    # on this exact fixture (an east-rising ramp, which LENGTHENS columns)
    # the ratio moves the safe direction, 0.5000 -> 0.4990. The mirror case
    # (west-rising, which shortens them) measured 0.5010, unchanged. Both
    # numbers are recorded here so nobody later tunes a threshold that was
    # never at risk.
    assert unpainted / total < 0.55, f"{unpainted}/{total} unpainted pixels in the outline bbox"


def test_slope_shade_is_gathered_through_uv_idx():
    """_render_tile_sloped must gather _slope_shade through
    sloped_quad_indices' uv_idx, never slice it to length.

    The fixture is chosen to make the bug SILENT, which is the only way to
    pin it: at tile_px=32 with corners (8, 0, 0, 0) the resample emits
    exactly 256 pixels, the same as diamond_indices(32), so
    `shade[:dst_y.size]` neither raises nor changes coverage -- it just
    shades the wrong pixels. (The corner choice is load-bearing, not
    arbitrary: (0, 8, 0, 0) emits 384 and would make the wrong version
    crash instead, which tests nothing.) The twist term is non-zero, so
    shading genuinely varies across the tile and the two candidates really
    do differ.

    A deterministic synthetic texture is patched in rather than relying on
    whatever the machine's configured game install returns, so the assert
    is on exact bytes rather than on a fallback color path."""
    tile_px = 32
    corners = (8, 0, 0, 0)
    dst_y, dst_x, src_y, src_x, uv_idx = ig.sloped_quad_indices(tile_px, *corners)
    assert dst_y.size == ig.diamond_indices(tile_px)[0].size, (
        "fixture no longer has the silent-failure shape -- pick corners whose run lengths still sum to a diamond"
    )

    w = h = 3
    proj = ig.canvas_size_and_origin(w, h, tile_px, 0, ig.MAX_ELEVATION, corner_headroom_steps=1)
    corner_rise = np.zeros((h + 1, w + 1), dtype=np.int64)
    corner_rise[1, 1] = corners[0]  # tile (1, 1)'s NW corner; its other three stay 0

    shade = render_mod._slope_shade(tile_px, *corners, proj.elev_step)
    gathered = shade[uv_idx]
    sliced = shade[: dst_y.size]
    assert not np.array_equal(gathered, sliced), "fixture no longer distinguishes the two candidates"

    rng = np.random.default_rng(3)
    texture = rng.integers(0, 256, size=(tile_px, tile_px, 3), dtype=np.uint8)
    real = render_mod.asset_source.get_terrain_texture_array
    render_mod.asset_source.get_terrain_texture_array = lambda _tid: texture
    try:
        img = np.zeros((proj.canvas_h, proj.canvas_w, 3), dtype=np.uint8)
        tile = SimpleNamespace(x=1, y=1, terrain_id=0, elevation=0)
        render_mod._render_tile_sloped(img, tile, tile_px, proj, corner_rise)
    finally:
        render_mod.asset_source.get_terrain_texture_array = real

    ox, oy = render_mod._crop_offset(tile.x, tile.y, texture.shape[0], tile_px)
    top = texture[oy : oy + tile_px, ox : ox + tile_px][src_y, src_x]
    expected = np.clip(top.astype(np.float32) * gathered[:, None], 0, 255).astype(np.uint8)
    wrong = np.clip(top.astype(np.float32) * sliced[:, None], 0, 255).astype(np.uint8)
    assert not np.array_equal(expected, wrong), "the two candidates must be distinguishable in pixels too"

    base_x, base_y = ig.tile_screen_origin(tile.x, tile.y, 0, proj)
    base_y -= min(corners)
    ay, ax = base_y + dst_y, base_x + dst_x
    assert ay.min() >= 0 and ay.max() < proj.canvas_h and ax.min() >= 0 and ax.max() < proj.canvas_w
    assert np.array_equal(img[ay, ax], expected)


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


# ---------------------------------------------------------------------------
# The shading constants (2026-08-23 calibration).
#
# Before this, SLOPE_LIGHT_DIR / SLOPE_SHADE_STRENGTH / SLOPE_SHADE_MIN /
# SLOPE_SHADE_MAX were pinned by NO test at all: the flat-map oracle above is
# insensitive to every one of them by construction (at n == up the shade is
# exactly 1.0 for any light direction or strength, and 1.0 sits strictly
# inside the clamp), and the uv_idx test only compares two gathers of the same
# array. So all four could be changed and the whole suite stayed green --
# exactly the hazard test_sloped_geometry's corner-rule test was written to
# close, still open here until now.
#
# These follow that test's pattern: assert a falsifiable CONSEQUENCE of what
# the captures showed, rebuilt in-test from a shape named in the public
# tools/gen_elevation_reference.py, with the provenance in the docstring. No
# image fixture and no path into the private capture set.
# ---------------------------------------------------------------------------


def _shade_for(nw, ne, sw, se, tile_px=64, elev_step=16):
    """One tile's shade factors, corners given in ELEVATION LEVELS."""
    s = elev_step
    return render_mod._slope_shade(tile_px, nw * s, ne * s, sw * s, se * s, s)


def test_slope_light_shades_the_flank_the_captures_show_shaded():
    """Pins the sign of SLOPE_LIGHT_DIR's horizontal component.

    The shape is `ridge` from tools/gen_elevation_reference.py: a straight
    Delta=1 band raised along +mapx. Under the `max` corner rule that leaves
    the band flat on top and puts a one-tile ramp ring on each side -- the
    -mapy ramp facing screen upper-left, the +mapy ramp facing screen lower-
    right (iso_geometry: screen_x grows with mapx+mapy, screen_y with
    mapy-mapx, +y down).

    In the 2026-08-22 in-game captures DE darkens the -mapy flank and
    brightens the +mapy one, measured at 0.727 and 1.186 against flat ground
    by a least-squares fit, and independently at 0.68 / 1.16 by a
    perpendicular luminance profile across the same capture.

    The constants that shipped until 2026-08-23 had this exactly backwards
    (0.997 on the flank DE darkens, 0.824 on the one DE brightens). That is
    the regression this test exists to catch, so it asserts the ORDERING and
    the side, not just that some shading happens.
    """
    dark = _shade_for(0, 0, 1, 1)  # -mapy flank, screen upper-left
    light = _shade_for(1, 1, 0, 0)  # +mapy flank, screen lower-right

    assert dark.max() < 1.0, "the -mapy flank must be darkened, not lit"
    assert light.min() > 1.0, "the +mapy flank must be lit, not darkened"
    assert dark.max() < light.min()

    # Same again across the other axis, from `plateau`'s straight +mapx edge:
    # the captures put the lit side toward +mapx as well as +mapy. Note the
    # corner pattern reads "backwards" -- a tile whose height RISES toward
    # +mapx has a surface normal pointing -mapx, so (1, 0, 1, 0), the ramp
    # descending away from a raised block, is the one facing the light.
    assert _shade_for(1, 0, 1, 0).min() > 1.0  # +mapx-facing ramp, lit
    assert _shade_for(0, 1, 0, 1).max() < 1.0  # -mapx-facing ramp, darkened

    # Rough magnitude, generously bounded -- the point is that the lit side is
    # actually visible. The pre-calibration constants could only reach +4.6%.
    assert light.mean() > 1.10


def test_slope_shade_clamps_are_guard_rails_and_do_not_fire():
    """SLOPE_SHADE_MIN/MAX must stay OUTSIDE the reachable range.

    They are guard rails, not part of the calibration, and for a long time
    they were simply dead: at the old strength the model spanned only
    [0.637, 1.046] against clamps of [0.6, 1.25], so tuning either one changed
    nothing on any map. At the calibrated strength the range is much wider and
    the rails are much closer to live, so this pins that they still clear
    every corner configuration -- if a future strength change makes a clamp
    bite, the shading silently stops being what was measured.
    """
    import itertools

    lo, hi = 1.0, 1.0
    for cfg in itertools.product((0, 1, 2), repeat=4):
        s = _shade_for(*cfg)
        lo, hi = min(lo, float(s.min())), max(hi, float(s.max()))

    assert lo > render_mod.SLOPE_SHADE_MIN, f"MIN bites: reachable low is {lo}"
    assert hi < render_mod.SLOPE_SHADE_MAX, f"MAX bites: reachable high is {hi}"
    # The measured range at the shipped constants, so a drift is visible here
    # rather than only in the clamp assertions above.
    assert lo == pytest.approx(0.5305, abs=0.01)
    assert hi == pytest.approx(1.2755, abs=0.01)


@pytest.mark.parametrize("elev_step_pct", [25, 50, 100, 150, 200])
def test_slope_shade_is_invariant_to_elev_step(elev_step_pct):
    """Shading does not depend on the elevation step at all.

    _slope_shade normalizes the surface gradient by elev_step, so a tile with
    the same corner configuration in LEVELS shades identically at every
    elev_step_pct stop. Worth pinning because it is the reason the 2026-08-23
    calibration is independent of the separate (still open) question of
    whether DEscape should adopt DE's own step height: changing that cannot
    disturb these constants.
    """
    half_h = 16
    step = max(1, round(half_h * elev_step_pct / 100))
    got = render_mod._slope_shade(64, 0, 0, step, step, step)
    ref = render_mod._slope_shade(64, 0, 0, 16, 16, 16)
    assert np.allclose(got, ref)
