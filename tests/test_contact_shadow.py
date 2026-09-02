"""Verifies the v2.6 Stepped contact-shadow band is a WEDGE confined to the
back neighbour whose elevation delta produced it, and the regression bar
for the spill bug this fixes (measured 61.3% of darkened pixels on target
at tile_px=64 before the fix, 100% after).

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
# Elevation difference between adjacent terraces in the fixture -- see
# _pyramid_scenario() for why 1 alone is not enough coverage.
STEPS = [1, 2]


def _pyramid_scenario(step: int = 1):
    """A small 4-terrace pyramid centred on (10, 10), built by DIRECT
    tile.elevation assignment -- never set_tiles_elevation, whose
    propagation would make the fixture's own shape depend on brush
    behaviour instead of being deterministic here.

    step is the elevation difference between adjacent terraces. step=1
    alone is NOT enough coverage: it makes every band's rise_px exactly
    proj.elev_step, so nothing with delta > 1 is ever tested -- and while
    the Set Elevation brush's propagation does hold neighbours to +-1, a
    scenario authored in DE (or raised with the Elevate tool, which has no
    upper clamp) can differ by more between adjacent tiles.
    """
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        rings = max(0, 3 - max(abs(tile.x - 10), abs(tile.y - 10)))
        tile.elevation = min(ig.MAX_ELEVATION, step * rings)
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
@pytest.mark.parametrize("step", STEPS)
def test_band_never_leaves_its_own_back_neighbour(tile_px, elev_step_pct, step):
    """THE regression assertion. Re-expresses every band pixel in the back
    NEIGHBOUR's own local coordinates and requires it to land inside that
    neighbour's diamond stencil. Pure geometry, no monkeypatching of
    private render symbols.
    """
    half_w, half_h = ig.half_dims(tile_px)
    dy, dx, _, _ = ig.diamond_indices(tile_px)
    mask = np.zeros((2 * half_h, 2 * half_w), dtype=bool)
    mask[dy, dx] = True

    scenario = _pyramid_scenario(step)
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

    # Emptiness is derived, not hardcoded to pct=200: with step > 1 a band
    # goes empty at a lower pct too, since rise_px scales with the delta.
    proj = ig.canvas_size_and_origin(
        scenario.map_manager.map_width,
        scenario.map_manager.map_height,
        tile_px,
        ig.MIN_ELEVATION,
        ig.MAX_ELEVATION,
        elev_step_pct=elev_step_pct,
    )
    if step * proj.elev_step >= 2 * half_h - 2:
        assert n_pixels == 0, "the caster fully hides its neighbour, so every band should be empty"
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
@pytest.mark.parametrize("step", STEPS)
def test_no_pixel_darkened_twice_across_the_whole_map(tile_px, elev_step_pct, step):
    """A pixel darkened by two different bands would compound CONTACT_SHADE
    into a much darker artifact than either band intends -- and is exactly
    what the old apex overshoot produced where two casters' bands crossed.
    """
    scenario = _pyramid_scenario(step)
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
            assert (factors <= 1.0).all()
            # Darkest exactly at the contact row.
            assert np.allclose(factors[depth == 0], render.CONTACT_SHADE)


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_falloff_ramps_out_within_a_fixed_fraction_of_a_tile(tile_px):
    """The band's WEIGHT is capped near the silhouette even though its
    EXTENT is still the whole exposed sliver -- the fix for "a one-level
    step darkens 51.6% of the neighbour tile at elev_step_pct=50, 82.0% at
    10", which rendered as a lattice of dark triangles instead of relief.

    Guards the property, not the constant: any pixel at or beyond the ramp
    must be exactly 1.0 (an exact no-op multiply in _clipped_darken, so
    the tail costs nothing visible), and every column must reach 1.0
    somewhere -- including the short columns near the apex, where the ramp
    compresses into the sliver rather than truncating mid-gradient. That
    last half is what the inclusive `eff - 1` endpoint buys; the exclusive
    `depth / span` it replaced left a 3-row apex column ending at 0.88.
    """
    _half_w, half_h = ig.half_dims(tile_px)
    ramp = max(1, half_h // render.CONTACT_RAMP_DIVISOR)
    for side in ("up_left", "up_right"):
        for rise_px in (1, max(1, half_h // 2), 2 * half_h - 3):
            _dy, dst_x, depth, span = ig.shadow_quad_indices(tile_px, rise_px, side)
            if depth.size == 0:
                continue
            factors = render._shadow_factors(tile_px, rise_px, side)
            where = f"tile_px={tile_px} rise_px={rise_px} side={side}"
            denom = np.maximum(np.minimum(span, ramp) - 1, 1)
            assert (factors[depth >= denom] == 1.0).all(), f"{where}: still darkening past the ramp"
            assert (factors[depth < denom] < 1.0).all(), f"{where}: no darkening inside the ramp"
            # Every column fades out, so no column ends on a hard edge --
            # except a 1px apex column, which is its own contact row.
            for col in np.unique(dst_x):
                sel = dst_x == col
                if span[sel][0] == 1:
                    continue
                assert factors[sel].max() == 1.0, f"{where} col={col}: column never reaches 1.0"
            # And the darkening is confined: at the widest column the ramp
            # is a fixed fraction of a tile, never the whole sliver.
            if span.max() > ramp:
                assert (factors < 1.0).sum() < depth.size, f"{where}: whole band darkened"


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


# --- The apex wedge: the band's two flanks cannot reach the columns where
# the DIAGONAL back neighbour (x+1, y-1) is what shows above the caster's
# top edge, so a run of adjacent casters read as separate blocks rather
# than one terrace-long band. See iso_geometry.shadow_apex_indices.
#
# Note what these do NOT do: none of them is a connectivity proof in two
# dimensions. Column coverage is what the original repro script measured,
# and it is exactly why that script's "2px hole" model pointed at the wrong
# fix -- adjacent casters sit half_h apart vertically, so sharing a column
# does not put pixels in adjacent rows. The render is the evidence for
# connectedness; these pin the contract the render relies on.


def _apex_and_band_columns(tile_px, rise_px):
    _ay, ax, _ad, _asp = ig.shadow_apex_indices(tile_px, rise_px)
    band = set()
    for side in ("up_left", "up_right"):
        _dy, dx, _d, _sp = ig.shadow_quad_indices(tile_px, rise_px, side)
        band.update(int(c) for c in np.unique(dx))
    return {int(c) for c in np.unique(ax)}, band


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_apex_wedge_and_band_never_share_a_column(tile_px):
    """The two passes must partition, not overlap. A shared column would
    darken twice, squaring CONTACT_SHADE on the tile's most visible
    column -- the exact hazard shadow_quad_indices' strict up_left/up_right
    split exists to prevent, reintroduced one level up."""
    _half_w, half_h = ig.half_dims(tile_px)
    for rise_px in range(1, 2 * half_h):
        apex, band = _apex_and_band_columns(tile_px, rise_px)
        assert not (apex & band), f"tile_px={tile_px} rise_px={rise_px} overlap {sorted(apex & band)}"


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_apex_wedge_fills_every_column_the_band_leaves_bare(tile_px):
    """The regression bar for the connect fix. Together the band and the
    wedge must claim EVERY used column of the diamond, with no bare column
    left between them.

    An earlier attempt used a strict `2*tops[c] < rise_px` adjacency test
    and left two bare columns on each flank of the wedge (measured at
    tile_px=64, rise_px=8: columns 23/24 and 39/40), which is a visible
    break in the contour. The bound is inclusive for that reason."""
    _tops, _bottoms, used = ig._diamond_column_edges(tile_px)
    expected = {int(c) for c in np.flatnonzero(used)}
    _half_w, half_h = ig.half_dims(tile_px)
    for rise_px in range(1, 2 * half_h - 2):
        apex, band = _apex_and_band_columns(tile_px, rise_px)
        missing = expected - (apex | band)
        assert not missing, f"tile_px={tile_px} rise_px={rise_px} bare columns {sorted(missing)}"


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_apex_wedge_stays_inside_the_diagonal_neighbours_own_diamond(tile_px):
    """The flat-ground-spill guard for this pass, and the reason it is safe
    to darken a tile nothing else touches.

    check_shadow_geometry guards the BAND against spill by asserting it
    equals its neighbour's exposed sliver exactly. The wedge needs its own
    equivalent: every pixel must land inside the diagonal neighbour's
    diamond, which sits at screen offset (0, -2*half_h + rise_px). If this
    ever fails, the pass is painting shadow onto whatever is behind the
    diagonal -- the v0.2 spill bug's signature, one tile further out."""
    tops, bottoms, _used = ig._diamond_column_edges(tile_px)
    _half_w, half_h = ig.half_dims(tile_px)
    for rise_px in range(1, 2 * half_h):
        dst_y, dst_x, _d, _s = ig.shadow_apex_indices(tile_px, rise_px)
        if dst_y.size == 0:
            continue
        diag_top = tops[dst_x] - 2 * half_h + rise_px
        diag_bottom = bottoms[dst_x] - 2 * half_h + rise_px
        assert np.all(dst_y >= diag_top), f"tile_px={tile_px} rise_px={rise_px} above the diagonal"
        assert np.all(dst_y <= diag_bottom), f"tile_px={tile_px} rise_px={rise_px} below the diagonal"
        # And never onto the caster's own top face, which is painted after.
        assert np.all(dst_y < tops[dst_x]), f"tile_px={tile_px} rise_px={rise_px} on the caster"


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_apex_wedge_empties_exactly_when_the_diagonal_is_hidden(tile_px):
    """Empty iff rise_px >= 2*half_h. Deliberately a LATER threshold than
    the band's own 2*half_h - 2: the diagonal is a further half_h up-screen
    and stays visible slightly longer, so at the tallest elev_step_pct
    stops this draws a thin cue where the band draws nothing at all."""
    _half_w, half_h = ig.half_dims(tile_px)
    for rise_px in range(1, 3 * half_h):
        dst_y, _dx, _d, _s = ig.shadow_apex_indices(tile_px, rise_px)
        if rise_px >= 2 * half_h:
            assert dst_y.size == 0, f"tile_px={tile_px} rise_px={rise_px} should be empty"
        else:
            assert dst_y.size > 0, f"tile_px={tile_px} rise_px={rise_px} should be non-empty"


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_apex_wedge_depth_and_span_contract(tile_px):
    """0 <= depth < span elementwise, the same contract shadow_quad_indices
    states, since _shadow_factors divides by span-1 and would otherwise
    produce a factor outside [CONTACT_SHADE, 1]."""
    _half_w, half_h = ig.half_dims(tile_px)
    for rise_px in (1, half_h, 2 * half_h - 1):
        dst_y, dst_x, depth, span = ig.shadow_apex_indices(tile_px, rise_px)
        if dst_y.size == 0:
            continue
        assert dst_y.shape == dst_x.shape == depth.shape == span.shape
        assert np.all(depth >= 0) and np.all(depth < span)
        factors = render._shadow_factors(tile_px, rise_px, "apex")
        assert factors.dtype == np.float32 and factors.shape == dst_y.shape
        assert np.all(factors >= render.CONTACT_SHADE) and np.all(factors <= 1.0)


def test_apex_wedge_rejects_a_nonpositive_rise():
    with pytest.raises(ValueError):
        ig.shadow_apex_indices(64, 0)


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_apex_wedge_is_thinner_than_the_band_it_bridges(tile_px):
    """The apex side ramps over HALF the band's fraction of half_h. With a
    shared ramp the wedge renders at the band's full thickness while the
    band it joins has already tapered to half that at the junction, and the
    step reads as a horizontal barb off every tile apex (seen directly in
    the A/B render before this was added)."""
    _half_w, half_h = ig.half_dims(tile_px)
    rise_px = max(1, half_h // 2)
    apex = render._shadow_factors(tile_px, rise_px, "apex")
    darkened = int(np.count_nonzero(apex < 1.0))
    _dy, _dx, _d, span = ig.shadow_apex_indices(tile_px, rise_px)
    per_column = darkened / max(1, len(np.unique(_dx)))
    full_ramp = max(1, half_h // render.CONTACT_RAMP_DIVISOR)
    assert per_column <= full_ramp, f"apex ramp {per_column} not thinner than band ramp {full_ramp}"


def _level_diagonal_scenario(anchor=(60, 60)):
    """Whole map at elevation 1 except the caster's two DIRECT back
    neighbours, dropped to 0. So the caster qualifies (both direct back
    neighbours are lower) while its DIAGONAL (x+1, y-1) is LEVEL with it.

    That combination is the apex wedge's only real hazard: the wedge paints
    onto the diagonal, so if it ignores the diagonal's own elevation it
    draws a shadow across ground at the caster's own height. That is the
    v0.2 spill bug's exact shape, one tile further out."""
    cx, cy = anchor
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 1
    for tx, ty in ((cx, cy - 1), (cx + 1, cy)):
        for tile in scenario.map_manager.terrain:
            if (tile.x, tile.y) == (tx, ty):
                tile.elevation = 0
    return scenario


def _flat_at_one():
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 1
    return scenario


def test_apex_wedge_never_darkens_a_diagonal_level_with_the_caster():
    """Render-level, and it has to be: the diagonal's elevation is checked
    in _render_tile_iso, not in the geometry, so a geometry-only assertion
    cannot see this gate being dropped. Mutation-checked -- replacing the
    `delta > 0` diagonal test with an unconditional draw turns this red,
    and turns NOTHING else in the suite red.

    Compares the diagonal tile's own diamond pixels against a fully flat
    control at the same elevation. Nothing legitimately shades that tile
    here: the only two tiles it is a back neighbour of are the two dropped
    ones, and they are LOWER than it, so neither casts."""
    from descape import settings

    anchor = (60, 60)
    diag = (anchor[0] + 1, anchor[1] - 1)

    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
    settings.set_elev_step_pct(50)

    img, _elev, proj = render.render_terrain_iso_with_proj(_level_diagonal_scenario(anchor), with_units=False)
    ctrl, _celev, cproj = render.render_terrain_iso_with_proj(_flat_at_one(), with_units=False)
    assert (proj.tile_px, proj.elev_step) == (cproj.tile_px, cproj.elev_step)

    dy, dx, _sy, _sx = ig.diamond_indices(proj.tile_px)
    bx, by = ig.tile_screen_origin(diag[0], diag[1], 1, proj)
    got = img[by + dy, bx + dx]
    want = ctrl[by + dy, bx + dx]

    differing = int(np.count_nonzero(np.any(got != want, axis=-1)))
    assert differing == 0, (
        f"{differing} pixels of the diagonal at {diag} were darkened even though it is "
        f"LEVEL with its caster -- the apex wedge is spilling onto flat ground"
    )
