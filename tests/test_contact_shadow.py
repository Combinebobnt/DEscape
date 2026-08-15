"""Verifies the v2.6 Stepped contact-shadow band is a WEDGE confined to the
back neighbour whose elevation delta produced it -- maintainer/docs/
PLAN_CONTACT_SHADOW.md's commit 3, and the regression bar for the spill bug
that plan fixes (measured 61.3% of darkened pixels on target at tile_px=64
before the fix, 100% after).

Root cause being guarded: shadow_quad_indices used to extrude a band of
CONSTANT height rise_px straight up from every column of the caster's own
diamond top edge. The strip of the back neighbour actually visible above
that edge is a wedge tapering to zero at the apex, so the old band
overshot near the apex onto tiles never tested for a delta -- usually the
diagonal (x+1, y-1), at the SAME elevation as the caster, i.e. a shadow
drawn across flat ground.

Deliberately NOT reading `settings`: the geometry tests take tile_px and
elev_step_pct as parameters and pass elev_step_pct= straight to
canvas_size_and_origin, so they never touch global graphics state at all.
That satisfies tests/README.md:143-147 (render output pinned to a stated
quality rather than "whatever the last developer saved") more strongly
than pinning would -- there is nothing global to pin. The ONE
render-touching test does pin both explicitly, since render_terrain_iso
resolves them from settings itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import iso_geometry as ig
from descape import render
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

TILE_PX = [16, 32, 64]
PCTS = [10, 50, 200]


def _pyramid_scenario():
    """A small 4-level pyramid centred on (10, 10), built by DIRECT
    tile.elevation assignment -- never set_tiles_elevation, whose
    propagation would make the fixture's own shape depend on brush
    behaviour instead of being deterministic here.
    """
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = max(0, 3 - max(abs(tile.x - 10), abs(tile.y - 10)))
    return scenario


def _elevations(scenario) -> np.ndarray:
    mm = scenario.map_manager
    elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
    for tile in mm.terrain:
        elevations[tile.y, tile.x] = tile.elevation
    return elevations


def _bands(scenario, tile_px, elev_step_pct):
    """Yields (caster, side, neighbour, base, neighbour_base, band) for every
    real shadow-casting (tile, side) pair, exactly as _render_tile_iso
    selects them.
    """
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    elevations = _elevations(scenario)
    proj = ig.canvas_size_and_origin(
        w, h, tile_px, ig.MIN_ELEVATION, ig.MAX_ELEVATION, elev_step_pct=elev_step_pct
    )
    for y in range(h):
        for x in range(w):
            e = int(elevations[y, x])
            for side, nx, ny in (("up_left", x, y - 1), ("up_right", x + 1, y)):
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                delta = e - int(elevations[ny, nx])
                if delta <= 0:
                    continue
                rise_px = delta * proj.elev_step
                band = ig.shadow_quad_indices(tile_px, rise_px, side)
                yield (
                    (x, y, e),
                    side,
                    (nx, ny, int(elevations[ny, nx])),
                    ig.tile_screen_origin(x, y, e, proj),
                    ig.tile_screen_origin(nx, ny, int(elevations[ny, nx]), proj),
                    band,
                )


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("elev_step_pct", PCTS)
def test_band_never_leaves_its_own_back_neighbour(tile_px, elev_step_pct):
    """THE regression assertion. Re-expresses every band pixel in the back
    NEIGHBOUR's own local coordinates and requires it to land inside that
    neighbour's diamond stencil. Pure geometry, no monkeypatching of
    private render symbols.
    """
    half_w, half_h = ig.half_dims(tile_px)
    dy, dx, _, _ = ig.diamond_indices(tile_px)
    mask = np.zeros((2 * half_h, 2 * half_w), dtype=bool)
    mask[dy, dx] = True

    scenario = _pyramid_scenario()
    n_pixels = 0
    for caster, side, neighbour, base, nbase, band in _bands(scenario, tile_px, elev_step_pct):
        s_dst_y, s_dst_x, _depth, _span = band
        if s_dst_y.size == 0:
            continue
        bx, by = base
        nbx, nby = nbase
        ly = by + s_dst_y - nby
        lx = bx + s_dst_x - nbx
        where = f"caster={caster} side={side} neighbour={neighbour}"
        assert ((ly >= 0) & (ly < 2 * half_h) & (lx >= 0) & (lx < 2 * half_w)).all(), (
            f"{where}: band left the neighbour's own bounding box"
        )
        assert mask[ly, lx].all(), f"{where}: band pixel outside the neighbour's diamond"
        n_pixels += s_dst_y.size

    if elev_step_pct == 200:
        assert n_pixels == 0, "every band should be empty at pct=200"
    else:
        assert n_pixels > 0, "fixture cast no shadow at all -- test would be vacuous"


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("rise_px_steps", [1, 2, 3])
def test_band_equals_the_whole_exposed_sliver(tile_px, rise_px_steps):
    """Completeness half of the plan's decision 1: the band is not merely
    INSIDE the neighbour, it is exactly the neighbour's whole exposed
    sliver -- every neighbour pixel visible above the caster's own
    silhouette on that side, no more and no less.
    """
    half_w, half_h = ig.half_dims(tile_px)
    tops, _bottoms, used = ig._diamond_column_edges(tile_px)
    dy, dx, _, _ = ig.diamond_indices(tile_px)
    diamond = set(zip(dy.tolist(), dx.tolist()))
    rise_px = rise_px_steps * max(1, half_h // 4)

    for side in ("up_left", "up_right"):
        off_x = -half_w if side == "up_left" else half_w
        s_dst_y, s_dst_x, _depth, _span = ig.shadow_quad_indices(tile_px, rise_px, side)
        got = set(zip(s_dst_y.tolist(), s_dst_x.tolist()))
        # The neighbour's diamond, expressed in the CASTER's local frame.
        neighbour = {(r - half_h + rise_px, c + off_x) for r, c in diamond}
        exposed = {
            (r, c)
            for (r, c) in neighbour
            if 0 <= c < 2 * half_w
            and used[c]
            and (c < half_w) == (side == "up_left")
            and r < tops[c]
        }
        assert got == exposed, (
            f"tile_px={tile_px} rise_px={rise_px} side={side}: "
            f"band != exposed sliver (extra={sorted(got - exposed)[:5]}, "
            f"missing={sorted(exposed - got)[:5]})"
        )


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("elev_step_pct", PCTS)
def test_no_pixel_darkened_twice_across_the_whole_map(tile_px, elev_step_pct):
    """A pixel darkened by two different bands would compound CONTACT_SHADE
    into a much darker artifact than either band intends -- and is exactly
    what the old apex overshoot produced where two casters' bands crossed.
    """
    scenario = _pyramid_scenario()
    seen: set = set()
    for _caster, _side, _neighbour, base, _nbase, band in _bands(scenario, tile_px, elev_step_pct):
        s_dst_y, s_dst_x, _depth, _span = band
        if s_dst_y.size == 0:
            continue
        bx, by = base
        pixels = set(zip((by + s_dst_y).tolist(), (bx + s_dst_x).tolist()))
        assert not (pixels & seen), f"{len(pixels & seen)} canvas pixels darkened by two bands"
        seen |= pixels


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_empty_band_has_correct_dtypes(tile_px):
    """Decision 2: when the caster fully hides its neighbour the band is
    empty and nothing is drawn -- no 1px floor, no clamp. All four arrays
    must still be int64 so downstream arithmetic doesn't change dtype on
    the empty path (the cumsum(avail) - avail form exists precisely so
    this falls out with no branch).
    """
    _half_w, half_h = ig.half_dims(tile_px)
    threshold = 2 * half_h - 2
    for side in ("up_left", "up_right"):
        arrays = ig.shadow_quad_indices(tile_px, threshold, side)
        assert len(arrays) == 4
        for arr in arrays:
            assert arr.size == 0
            assert arr.dtype == np.int64
        # And non-empty immediately below the threshold.
        below = ig.shadow_quad_indices(tile_px, threshold - 1, side)
        assert below[0].size > 0, f"tile_px={tile_px} side={side}: empty one step below threshold"


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_shadow_factors_dtype_and_range(tile_px):
    """float32, not float64 -- catches the int64 promotion trap: depth is
    int64, so a bare depth.astype(float32) / span would promote back to
    float64 and break the documented full-canvas/scratch-canvas bit
    identity.
    """
    _half_w, half_h = ig.half_dims(tile_px)
    for side in ("up_left", "up_right"):
        for rise_px in (1, max(1, half_h // 2), 2 * half_h - 3):
            factors = render._shadow_factors(tile_px, rise_px, side)
            _dst_y, _dst_x, depth, _span = ig.shadow_quad_indices(tile_px, rise_px, side)
            assert factors.dtype == np.float32
            assert factors.shape == depth.shape
            if factors.size == 0:
                continue
            assert (factors >= render.CONTACT_SHADE).all()
            assert (factors < 1.0).all()
            # Darkest exactly at the contact row.
            assert np.allclose(factors[depth == 0], render.CONTACT_SHADE)


def test_hill_renders_differently_from_flat_and_never_raises():
    """The one render-touching test -- pins graphics_quality/elev_step_pct
    explicitly, since render_terrain_iso resolves both from settings
    itself rather than taking them as parameters.
    """
    from descape import settings

    hill = _pyramid_scenario()
    flat = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in flat.map_manager.terrain:
        tile.elevation = 0

    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
    settings.set_elev_step_pct(50)
    hill_img = render.render_terrain_iso(hill, with_units=False)
    flat_img = render.render_terrain_iso(flat, with_units=False)
    assert hill_img.shape == flat_img.shape
    assert not np.array_equal(hill_img, flat_img), "a 4-level pyramid rendered identically to flat ground"

    for pct in (50, 200):
        settings.set_elev_step_pct(pct)
        img = render.render_terrain_iso(hill, with_units=False)
        assert img.dtype == np.uint8 and img.ndim == 3
