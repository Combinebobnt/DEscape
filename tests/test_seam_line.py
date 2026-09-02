"""Verifies the v2.6 Stepped SEAM LINE.

What it is: a 1px contour on a tile's OWN two up-screen diamond edges,
drawn when the neighbour behind that edge is lower. The contact-shadow
band it complements is drawn on the NEIGHBOUR's pixels, and the two tiles'
back edges converge at the caster's apex, so the band tapers to nothing
there -- measured coverage of a tile's back edge at tile_px=64, one level:
83.9% at elev_step_pct=25, 71.0% at 50, 45.2% at 100, 0% at 200. Each
terrace edge therefore read as a DASHED line ("thin strips and only
halfway", reported in-app 2026-08-16). The seam's extent is a property of
the caster alone, so it cannot taper or gap.

Honest scope note, because the file's most important-looking assertion is
weaker than it looks: test_covers_every_column_on_its_side pins the new
function's CONTRACT, it is not a regression bar. It "fails on today's
code" only in the sense that seam_edge_indices did not exist before. The
user-facing property -- a contour that stays continuous to the apex on
screen -- was confirmed by A/B render and eyeball, and nothing in this
file mechanically guards it.

Deliberately NOT reading `settings` for the geometry tests: they take
tile_px/elev_step_pct as parameters and pass elev_step_pct= straight to
canvas_size_and_origin, so there is no global graphics state to pin (the
same approach test_contact_shadow.py documents at more length). The one
render-touching test pins both explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import iso_geometry as ig
from descape import render
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

TILE_PX = [8, 16, 32, 64, 128]
SIDES = ["up_left", "up_right"]


def _flat_scenario():
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 0
    return scenario


def _pyramid_scenario(step: int = 1):
    """A 4-terrace pyramid centred on (10, 10), by DIRECT tile.elevation
    assignment -- never set_tiles_elevation, whose propagation would make
    the fixture's shape depend on brush behaviour. Mirrors
    test_contact_shadow.py's own fixture so the two files describe the same
    terrain.
    """
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        rings = max(0, 3 - max(abs(tile.x - 10), abs(tile.y - 10)))
        tile.elevation = min(ig.MAX_ELEVATION, step * rings)
    return scenario


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("side", SIDES)
def test_seam_is_a_subset_of_the_tiles_own_diamond(tile_px, side):
    """The seam darkens the CASTER, not its neighbour -- the one structural
    difference from the band, and what makes it safe to draw right after
    the top face and impossible for it to darken a unit (a tile's units are
    drawn after its own terrain).
    """
    dst_y, dst_x = ig.seam_edge_indices(tile_px, side)
    dy, dx, _src_y, _src_x = ig.diamond_indices(tile_px)
    diamond = set(zip(dy.tolist(), dx.tolist()))
    got = set(zip(dst_y.tolist(), dst_x.tolist()))
    assert got, f"tile_px={tile_px} side={side}: empty seam"
    assert got <= diamond, f"tile_px={tile_px} side={side}: {len(got - diamond)} seam px off the diamond"


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("side", SIDES)
def test_covers_every_column_on_its_side(tile_px, side):
    """Contract of the new function: exactly one pixel per used column on
    that side, at that column's own topmost diamond row -- no gaps, no
    duplicates. This is the property that makes the line continuous by
    construction rather than dependent on a neighbour's elevation.

    The two apex columns (half_w - 1, half_w) are excluded from BOTH
    sides as of 2026-08-16 -- seam_apex_indices owns them, see
    test_a_terrace_run_is_one_unbroken_column_span below for why.

    Not a regression bar -- see this module's docstring.
    """
    half_w, _half_h = ig.half_dims(tile_px)
    tops, _bottoms, used = ig._diamond_column_edges(tile_px)
    cols = np.arange(2 * half_w)
    in_side = (cols < half_w - 1) if side == "up_left" else (cols > half_w)
    expected_cols = cols[used & in_side]

    dst_y, dst_x = ig.seam_edge_indices(tile_px, side)
    assert np.array_equal(np.sort(dst_x), expected_cols), (
        f"tile_px={tile_px} side={side}: seam columns != that side's used columns"
    )
    assert dst_x.size == np.unique(dst_x).size, "a column got two seam pixels"
    assert np.array_equal(dst_y, tops[dst_x]), "a seam pixel is not on its column's top diamond row"
    assert dst_y.dtype == np.int64 and dst_x.dtype == np.int64


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_the_three_passes_partition_the_top_edge(tile_px):
    """Catches copying skirt_quad_indices' column split, which deliberately
    SHARES the apex column between its two sides. Right for a skirt (both
    sides really do reach the same bottom tip), wrong here: a shared column
    would be darkened twice whenever both back neighbours are lower --
    SEAM_SHADE squared on the one column most visible to the eye.

    Three passes since 2026-08-16, not two: the apex pair was split out of
    the sides to close the cross-tile gap. The partition property is what
    survived that change unweakened -- pairwise disjoint AND exhaustive,
    so the apex is still drawn exactly once no matter how many sides
    qualify.
    """
    half_w, _half_h = ig.half_dims(tile_px)
    left = set(zip(*(a.tolist() for a in ig.seam_edge_indices(tile_px, "up_left"))))
    right = set(zip(*(a.tolist() for a in ig.seam_edge_indices(tile_px, "up_right"))))
    apex = set(zip(*(a.tolist() for a in ig.seam_apex_indices(tile_px))))
    assert not (left & right), f"tile_px={tile_px}: {len(left & right)} px shared between the two sides"
    assert not (left & apex), f"tile_px={tile_px}: up_left overlaps the apex pass"
    assert not (right & apex), f"tile_px={tile_px}: up_right overlaps the apex pass"

    assert len(apex) == 2, f"tile_px={tile_px}: apex pass is {len(apex)} px, expected exactly 2"
    assert {x for _y, x in apex} == {half_w - 1, half_w}, "apex pass is not the two apex columns"

    _tops, _bottoms, used = ig._diamond_column_edges(tile_px)
    assert len(left) + len(right) + len(apex) == int(used.sum()), (
        "the three passes don't cover every used column"
    )


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("side", SIDES)
def test_never_touches_an_unused_column(tile_px, side):
    """Catches a dropped `used` mask. tops is argmax over the diamond
    membership mask and returns 0 for the two columns holding no diamond
    pixels at all (0 and 2*half_w - 1), so without the mask those two get a
    stray seam pixel at row 0 -- outside the diamond, on whatever tile is
    behind. Same trap shadow_quad_indices documents.
    """
    half_w, _half_h = ig.half_dims(tile_px)
    _dst_y, dst_x = ig.seam_edge_indices(tile_px, side)
    assert dst_x.min() >= 1, f"tile_px={tile_px} side={side}: seam on column 0"
    assert dst_x.max() <= 2 * half_w - 2, f"tile_px={tile_px} side={side}: seam on the last column"


def test_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="side must be"):
        ig.seam_edge_indices(64, "left")


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("side", SIDES)
def test_a_terrace_run_is_one_unbroken_column_span(tile_px, side):
    """THE regression bar for the 2026-08-16 apex fix, and the property
    nothing in this file guarded before it (see the module docstring's
    scope note, now discharged).

    A terrace edge is not drawn by one tile, it is drawn by a RUN of them,
    and adjacent tiles sit exactly half_w canvas px apart. Under the old
    strict partition a run where only ONE side qualified -- the common
    case, a straight terrace -- left a hole every half_w px, because
    up_left stopped at half_w - 1 and the next tile's up_left started at
    half_w + 1 in canvas terms. Measured on this file's own pyramid at
    pct=100: 558 seam px in 15 isolated components rather than continuous
    edges.

    Simulated in canvas columns rather than rendered, so it isolates the
    geometry from every rendering confound. Bounds come from half_dims and
    the run length, never a literal.
    """
    half_w, _half_h = ig.half_dims(tile_px)
    _side_y, side_x = ig.seam_edge_indices(tile_px, side)
    _apex_y, apex_x = ig.seam_apex_indices(tile_px)

    run = 6
    cols: set[int] = set()
    for k in range(run):
        # This side qualified, so this tile draws its side pass AND the
        # apex pass -- exactly what render._render_tile_iso does.
        cols |= {k * half_w + int(c) for c in side_x}
        cols |= {k * half_w + int(c) for c in apex_x}

    lo, hi = min(cols), max(cols)
    holes = sorted(c for c in range(lo, hi + 1) if c not in cols)
    assert not holes, (
        f"tile_px={tile_px} side={side}: {len(holes)} unpainted column(s) in a "
        f"{run}-tile terrace run, first few {holes[:8]} -- the seam is dashed, not continuous"
    )
    # And the run really spanned multiple tiles, so "no holes" isn't
    # vacuously true of a single tile's own contiguous columns.
    assert hi - lo >= (run - 1) * half_w, "the simulated run collapsed to one tile"


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("elev_step_pct", [25, 50, 100, 200])
def test_the_seam_span_is_pct_invariant_where_the_band_is_not(tile_px, elev_step_pct):
    """The original in-app complaint ("thin strips and only halfway") as a
    property rather than a threshold: the band's coverage of a tile's back
    edge collapses as elev_step_pct rises (83.9% at 25 down to 0% at 200,
    measured at tile_px=64), because its length depends on the NEIGHBOUR.
    The seam's does not depend on the neighbour at all, so its column set
    must be byte-identical across every pct -- including 200, where the
    band is empty and the seam is the only up-screen cue there is.
    """
    proj = ig.canvas_size_and_origin(
        8, 8, tile_px, ig.MIN_ELEVATION, ig.MAX_ELEVATION, elev_step_pct=elev_step_pct
    )
    rise_px = proj.elev_step
    if rise_px <= 0:
        pytest.skip("no rise at this tile_px/pct combination")

    _half_w, _half_h = ig.half_dims(tile_px)
    _tops, _bottoms, used = ig._diamond_column_edges(tile_px)
    covered = set()
    for side in SIDES:
        covered |= {int(c) for c in ig.seam_edge_indices(tile_px, side)[1]}
    covered |= {int(c) for c in ig.seam_apex_indices(tile_px)[1]}
    assert covered == set(np.flatnonzero(used).tolist()), (
        f"tile_px={tile_px} pct={elev_step_pct}: seam coverage changed with pct"
    )

    # The contrast that makes the assertion above meaningful: the band at
    # this same pct genuinely does not cover the edge.
    band_cols = set()
    for side in SIDES:
        band_cols |= {int(c) for c in ig.shadow_quad_indices(tile_px, rise_px, side)[1]}
    assert len(band_cols) <= len(covered), "fixture assumption broke: band wider than the diamond"
    if elev_step_pct == 200:
        assert not band_cols, "the band should be empty at pct=200 -- the seam-only case"


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("side", SIDES)
def test_seam_factors_dtype_and_alignment(tile_px, side):
    """_clipped_darken does factors[in_bounds] and factors[:, None], so a
    scalar breaks it; float64 breaks the documented full-canvas/scratch-
    canvas bit identity instead, silently.
    """
    factors = render._seam_factors(tile_px, side)
    dst_y, _dst_x = ig.seam_edge_indices(tile_px, side)
    assert factors.dtype == np.float32
    assert factors.ndim == 1
    assert factors.shape == dst_y.shape
    assert (factors > 0).all(), "a zero factor would make a painted pixel read as unpainted"
    assert (factors == np.float32(render.SEAM_SHADE)).all()


@pytest.mark.parametrize("tile_px", TILE_PX)
def test_seam_is_non_empty_where_the_band_is_empty(tile_px):
    """The elev_step_pct=200 residual this closes: once the caster fully
    hides its neighbour the band is correctly empty, and there is no skirt
    on the up-screen half either, so a hill had no up-screen elevation cue
    at all. The seam's extent doesn't depend on the neighbour, so it
    survives exactly where the band cannot.
    """
    _half_w, half_h = ig.half_dims(tile_px)
    rise_px = 2 * half_h - 2  # shadow_quad_indices' documented first-empty rise
    for side in SIDES:
        assert ig.shadow_quad_indices(tile_px, rise_px, side)[0].size == 0, "fixture assumption broke"
        assert ig.seam_edge_indices(tile_px, side)[0].size > 0, f"tile_px={tile_px} side={side}: seam empty too"


# The band cast by tile N onto its back neighbour lands on the OPPOSITE
# side of that neighbour's own diamond: an up_left band occupies caster
# columns c < half_w, which map to neighbour columns c + half_w, i.e. the
# neighbour's up_right half. Comparing a band against the neighbour's
# same-named seam finds nothing and the test passes vacuously -- that
# mistake is why this mapping is spelled out rather than inlined.
_OPPOSITE = {"up_left": "up_right", "up_right": "up_left"}


def _band_pixels_landing_on_the_neighbours_seam(tile_px, rise_px, side):
    """Band factors for exactly those band pixels that coincide with the
    back neighbour's own seam, in the NEIGHBOUR's local frame. Returns
    (all_factors_there, only_the_ones_still_darkening).
    """
    half_w, half_h = ig.half_dims(tile_px)
    s_dst_y, s_dst_x, _depth, _span = ig.shadow_quad_indices(tile_px, rise_px, side)
    if s_dst_y.size == 0:
        return [], []
    band_factors = render._shadow_factors(tile_px, rise_px, side)
    # The neighbour's diamond sits at (-half_w, -half_h + rise_px) from the
    # caster's for up_left, (+half_w, same) for up_right -- so invert that.
    off_x = half_w if side == "up_left" else -half_w
    nb_y = s_dst_y + half_h - rise_px
    nb_x = s_dst_x + off_x
    seam_y, seam_x = ig.seam_edge_indices(tile_px, _OPPOSITE[side])
    seam_px = set(zip(seam_y.tolist(), seam_x.tolist()))
    on_seam = [
        float(band_factors[i]) for i in range(nb_y.size) if (int(nb_y[i]), int(nb_x[i])) in seam_px
    ]
    return on_seam, [f for f in on_seam if f < 1.0]


@pytest.mark.parametrize("tile_px", TILE_PX)
@pytest.mark.parametrize("elev_step_pct", [25, 50, 100, 200])
def test_compounding_with_the_band_stays_bounded(tile_px, elev_step_pct):
    """The band drawn ONTO tile N reaches N's own top-edge row -- exactly
    where N's seam goes. Almost everywhere the band's factor there is
    exactly 1.0 (_shadow_factors' no-op endpoint), so a 1px seam does not
    compound at all; the exceptions are single-row (span == 1) columns
    near the apex, which carry the full CONTACT_SHADE.

    Measured across this whole grid, the exceptions are tile_px=16 at
    pct=25 and tile_px=8 at pct=50, both 2 pixels per band. (The plan
    swept tile_px 16/32/64 x pct 25/50/100 and so found only the first;
    tile_px=8 is exercised here as a standalone geometry value -- it is
    no longer reachable via the live mip ladder now that
    MIP_MIN_TILE_PIXELS is 16.) Worst case is
    SEAM_SHADE * CONTACT_SHADE -- accepted, and pinned here so it cannot
    silently become a general overlap if the seam thickens past 1px or
    the falloff ramp changes.

    The product must also stay well clear of 0: _clipped_darken's contract
    is that a painted pixel never becomes black, since
    verify_iso_render.check_full_coverage uses "black <=> unpainted" as
    its oracle.
    """
    proj = ig.canvas_size_and_origin(
        8, 8, tile_px, ig.MIN_ELEVATION, ig.MAX_ELEVATION, elev_step_pct=elev_step_pct
    )
    rise_px = proj.elev_step  # one level -- the shallowest, widest band
    if rise_px <= 0:
        pytest.skip("no rise at this tile_px/pct combination")
    for side in SIDES:
        _on_seam, compounding = _band_pixels_landing_on_the_neighbours_seam(tile_px, rise_px, side)
        where = f"tile_px={tile_px} pct={elev_step_pct} side={side}"
        assert len(compounding) <= 2, f"{where}: {len(compounding)} compounding px, expected at most 2"
        for f in compounding:
            assert f >= render.CONTACT_SHADE - 1e-6, f"{where}: band factor {f} below CONTACT_SHADE"
            product = f * render.SEAM_SHADE
            assert product > 0.3, f"{where}: compounded factor {product:.3f} is nearly black"


def test_the_compounding_probe_is_not_vacuous():
    """Guards the test above against silently passing because the frame
    mapping stopped finding anything -- the exact way its first draft was
    wrong. At least one configuration must really compound, and the two
    known ones must be found.
    """
    found = {}
    for tile_px in TILE_PX:
        for pct in (25, 50, 100, 200):
            proj = ig.canvas_size_and_origin(
                8, 8, tile_px, ig.MIN_ELEVATION, ig.MAX_ELEVATION, elev_step_pct=pct
            )
            if proj.elev_step <= 0:
                continue
            for side in SIDES:
                on_seam, compounding = _band_pixels_landing_on_the_neighbours_seam(
                    tile_px, proj.elev_step, side
                )
                assert on_seam or not ig.shadow_quad_indices(tile_px, proj.elev_step, side)[0].size, (
                    f"tile_px={tile_px} pct={pct} side={side}: a non-empty band met the seam nowhere, "
                    "so the frame mapping is wrong and the bound test is vacuous"
                )
                if compounding:
                    found[(tile_px, pct)] = len(compounding)
    assert (16, 25) in found and (8, 50) in found, f"known compounding configs missing: {found}"


def test_flat_ground_draws_no_seam_at_all():
    """Decision 3's whole point: without the per-side `delta > 0` test the
    seam becomes a permanent grid outline on flat terrain. Compares a flat
    render against one with SEAM_SHADE neutralised to an exact no-op --
    byte-identical means no seam ran anywhere.
    """
    from descape import settings

    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
    flat = _flat_scenario()

    original = render.SEAM_SHADE
    try:
        with_seam = render.render_terrain_iso(flat, with_units=False)
        render.SEAM_SHADE = 1.0  # exact no-op multiply in _clipped_darken
        render._seam_factors.cache_clear()
        without_seam = render.render_terrain_iso(flat, with_units=False)
    finally:
        render.SEAM_SHADE = original
        render._seam_factors.cache_clear()
    assert np.array_equal(with_seam, without_seam), "a flat map drew a seam somewhere"


def test_seam_changes_a_hill_at_every_pct_including_200():
    """The render-touching test. Pins graphics_quality explicitly (
    render_terrain_iso resolves it from settings), and drives elev_step_pct
    through the module global rather than settings.set_elev_step_pct --
    that setter PERSISTS to the user's config.yaml, which a test has no
    business doing.

    pct=200 is the case that matters: the band is empty there, so any
    difference at all is the seam's alone.
    """
    from descape import settings

    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
    hill = _pyramid_scenario()

    original_shade = render.SEAM_SHADE
    original_pct = settings._elev_step_pct
    shapes = set()
    try:
        for pct in (25, 50, 100, 200):
            settings._elev_step_pct = pct
            render.SEAM_SHADE = original_shade
            render._seam_factors.cache_clear()
            with_seam = render.render_terrain_iso(hill, with_units=False)
            shapes.add(with_seam.shape)
            render.SEAM_SHADE = 1.0
            render._seam_factors.cache_clear()
            without_seam = render.render_terrain_iso(hill, with_units=False)
            assert with_seam.dtype == np.uint8 and with_seam.ndim == 3
            changed = int(np.count_nonzero(np.any(with_seam != without_seam, axis=2)))
            assert changed > 0, f"pct={pct}: the seam changed nothing on a 4-level pyramid"
            # And it never drove a painted pixel to black -- the oracle
            # verify_iso_render.check_full_coverage depends on.
            newly_black = np.all(with_seam == 0, axis=2) & np.any(without_seam.astype(bool), axis=2)
            assert not newly_black.any(), f"pct={pct}: {int(newly_black.sum())} painted px turned black"
    finally:
        render.SEAM_SHADE = original_shade
        settings._elev_step_pct = original_pct
        render._seam_factors.cache_clear()

    # Without this the test passes whether or not the pct reached the
    # renderer at all: the seam draws at EVERY pct, so `changed > 0` four
    # times over is equally consistent with four identical renders. Canvas
    # height scales with elev_step, so four distinct shapes is proof the
    # loop really swept four configurations -- and in particular that the
    # pct=200 iteration, the one where the band is empty and the seam is
    # the only cue, was genuinely rendered at 200.
    assert len(shapes) == 4, f"elev_step_pct never reached the renderer -- got shapes {shapes}"


def _predicted_seam_pixels(scenario, elevations, proj):
    """Every pixel the seam SHOULD darken, rebuilt from the geometry
    functions and tile_screen_origin alone -- the render path's own
    qualification rule restated, never imported from it.
    """
    tile_px = proj.tile_px
    # Square-only, matching the renderer's own w/h: non-square (W x H) maps
    # are unsupported and deferred. If that ever lands, this helper
    # mispredicts silently rather than failing loudly.
    map_w = scenario.map_manager.map_size
    map_h = map_w
    predicted = set()
    for tile in scenario.map_manager.terrain:
        base_x, base_y = ig.tile_screen_origin(tile.x, tile.y, tile.elevation, proj)
        qualified = False
        for side, nx, ny in (("up_left", tile.x, tile.y - 1), ("up_right", tile.x + 1, tile.y)):
            if not (0 <= nx < map_w and 0 <= ny < map_h):
                continue
            if tile.elevation - int(elevations[ny, nx]) <= 0:
                continue
            qualified = True
            dy, dx = ig.seam_edge_indices(tile_px, side)
            predicted |= {(base_y + int(a), base_x + int(b)) for a, b in zip(dy.tolist(), dx.tolist())}
        if qualified:
            dy, dx = ig.seam_apex_indices(tile_px)
            predicted |= {(base_y + int(a), base_x + int(b)) for a, b in zip(dy.tolist(), dx.tolist())}
    return predicted


def test_rendered_seam_mask_equals_the_predicted_pixel_set():
    """The geometry <-> render link this module's docstring admitted was
    missing: every earlier test here checks index arrays, and the one
    render test only asserts "something changed". So the indices could be
    perfect and the render still draw them at the wrong origin, on the
    wrong tiles, or not at all.

    A/B against SEAM_SHADE = 1.0 isolates the seam exactly -- an exact
    no-op multiply, so any differing pixel is the seam's and nothing else
    (the band, skirts and top faces are byte-identical between the two).

    Not plain set equality: a predicted pixel over an already-black pixel
    cannot change (0 * 0.6 == 0), so those are excluded by checking the
    control render rather than by tolerating a fudge count.
    """
    from descape import settings

    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
    hill = _pyramid_scenario()

    original = render.SEAM_SHADE
    try:
        with_seam, elevations, proj = render.render_terrain_iso_with_proj(hill, with_units=False)
        render.SEAM_SHADE = 1.0
        render._seam_factors.cache_clear()
        without_seam, _e, _p = render.render_terrain_iso_with_proj(hill, with_units=False)
    finally:
        render.SEAM_SHADE = original
        render._seam_factors.cache_clear()

    ys, xs = np.nonzero(np.any(with_seam != without_seam, axis=2))
    changed = set(zip(ys.tolist(), xs.tolist()))
    predicted = _predicted_seam_pixels(hill, elevations, proj)
    assert predicted, "fixture drew no seam at all"

    stray = changed - predicted
    assert not stray, f"{len(stray)} pixel(s) changed outside the predicted seam, e.g. {sorted(stray)[:5]}"

    h, w = without_seam.shape[:2]
    missing = [
        (y, x)
        for (y, x) in (predicted - changed)
        if 0 <= y < h and 0 <= x < w and without_seam[y, x].any()
    ]
    assert not missing, (
        f"{len(missing)} predicted seam pixel(s) over non-black terrain never darkened, "
        f"e.g. {sorted(missing)[:5]}"
    )

    # Whole-canvas exactly-once, the strongest guard that the apex pass did
    # not make things worse than the gap it closed: the apex widens each
    # tile's own coverage, so if any tile's apex column landed on a
    # NEIGHBOUR's side column the two would compound to SEAM_SHADE**2. The
    # per-tile test below checks the both-sides-qualify case; this checks
    # every seam pixel on the canvas, cross-tile cases included, for free
    # from the two renders already taken.
    ys_a, xs_a = np.nonzero(np.any(with_seam != without_seam, axis=2))
    src = without_seam[ys_a, xs_a].astype(np.float32)
    once = np.clip(src * render.SEAM_SHADE, 0, 255).astype(np.uint8)
    not_once = ~np.all(with_seam[ys_a, xs_a] == once, axis=1)
    assert not not_once.any(), (
        f"{int(not_once.sum())} of {len(ys_a)} seam px are not exactly one SEAM_SHADE "
        "application -- something darkened them twice"
    )


def test_apex_columns_are_drawn_exactly_once():
    """Guards the double-darkening the three-way partition exists to
    prevent. On a pyramid's corner tile BOTH back neighbours are lower, so
    both sides qualify -- the exact case where letting each qualifying
    side extend into the apex (the obvious simpler fix, and the one this
    plan rejected) would apply SEAM_SHADE twice on the tile's single most
    visible column.

    Asserts the apex pixel equals ONE application of SEAM_SHADE, computed
    from the control render's own value at that pixel, so it pins the
    count rather than a colour constant.
    """
    from descape import settings

    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
    hill = _pyramid_scenario()

    original = render.SEAM_SHADE
    try:
        with_seam, elevations, proj = render.render_terrain_iso_with_proj(hill, with_units=False)
        render.SEAM_SHADE = 1.0
        render._seam_factors.cache_clear()
        without_seam, _e, _p = render.render_terrain_iso_with_proj(hill, with_units=False)
    finally:
        render.SEAM_SHADE = original
        render._seam_factors.cache_clear()

    tile_px = proj.tile_px
    apex_dy, apex_dx = ig.seam_apex_indices(tile_px)
    map_w = hill.map_manager.map_size

    checked = 0
    for tile in hill.map_manager.terrain:
        # Both back neighbours strictly lower -- both sides qualify.
        sides_lower = 0
        for nx, ny in ((tile.x, tile.y - 1), (tile.x + 1, tile.y)):
            if 0 <= nx < map_w and 0 <= ny < map_w and tile.elevation - int(elevations[ny, nx]) > 0:
                sides_lower += 1
        if sides_lower != 2:
            continue
        base_x, base_y = ig.tile_screen_origin(tile.x, tile.y, tile.elevation, proj)
        for a, b in zip(apex_dy.tolist(), apex_dx.tolist()):
            y, x = base_y + int(a), base_x + int(b)
            src = without_seam[y, x].astype(np.float32)
            if not src.any():
                continue  # black underneath -- darkening is unobservable
            once = np.clip(src * render.SEAM_SHADE, 0, 255).astype(np.uint8)
            got = with_seam[y, x]
            assert np.array_equal(got, once), (
                f"tile ({tile.x},{tile.y}) apex px ({y},{x}): got {got.tolist()}, "
                f"expected one SEAM_SHADE application {once.tolist()} "
                f"(twice would be {np.clip(src * render.SEAM_SHADE**2, 0, 255).astype(np.uint8).tolist()})"
            )
            checked += 1

    assert checked > 0, "no tile on the pyramid had both back neighbours lower -- test is vacuous"
