"""Track C5's Step 2: a Sloped unit is painted ON the surface the
compositor drew, at its own sub-tile point.

The one check that matters here compares the unit's ACTUALLY PAINTED pixels
against the terrain pixels sloped_quad_indices ACTUALLY produced -- never
against a second copy of _draw_unit_sloped's own base_y expression. That
distinction is the whole test: the silent-failure class Step 2 introduces is
a convention mismatch (unit_rise_px reports an ABSOLUTE rise, while
sloped_quad_indices' dst_y is relative to the tile's own d_min, which
_render_tile_sloped folds back in at its base_y), and a check written
against the same formula the renderer uses passes with that mismatch
present.

So: the unit side is MEASURED off the rendered image, and the terrain side
comes from sloped_quad_indices plus _render_tile_sloped's own separately
pinned placement convention. Two different derivations, and the fixture
asserts d_min > 0 so the mismatch has something to be wrong by.
"""

from __future__ import annotations


import numpy as np
import pytest

from testkit.fakes import (
    FakeScenario,
    SyntheticTile,
    SyntheticUnit,
)

from descape import iso_geometry as ig
from descape import render
from descape.terrain_palette import BUILDING_TILE_SPANS, PLAYER_COLORS

MAP_W = MAP_H = 16
_PLAYER = 1
# Single-tile so the marker is one diamond and the column measurement below
# is unambiguous; span looked up rather than hardcoded, the same idiom
# test_stepped_unit_fill.py uses for its building.
_UNIT_CONST = next(uid for uid, (sx, sy) in BUILDING_TILE_SPANS.items() if (sx, sy) == (1, 1))




def _ramp_scenario(unit_x: float, unit_y: float) -> FakeScenario:
    """A west-to-east ramp with one unit on it.

    Elevation rises with x, so a tile painted LATER in depth_order (larger
    y - x, or the same y - x further along x) is never both higher and
    overlapping -- which is what keeps the unit's own diamond unoccluded and
    the column measurement below readable. The run-length assertion makes a
    violation loud rather than silent if that ever stops holding.
    """
    units = [[] for _ in range(9)]
    units[_PLAYER] = [SyntheticUnit(x=unit_x, y=unit_y, unit_const=_UNIT_CONST)]
    tiles = [
        SyntheticTile(x=x, y=y, elevation=max(0, min(4, x - 6)))
        for y in range(MAP_H)
        for x in range(MAP_W)
    ]
    return FakeScenario(MAP_W, MAP_H, tiles, units)


def _diamond_pixel_nearest(tile_px: int, fx: float, fy: float) -> int:
    """Index into diamond_indices(tile_px) of the painted pixel whose own
    tile fractions are closest to (fx, fy) -- so the unit can be placed at
    EXACTLY a painted pixel's own sub-tile position and the comparison
    below is exact rather than nearest-neighbour."""
    fp, fq = ig.tile_uv_fractions(tile_px)
    pfx, pfy = 1 - fq, fp
    return int(np.argmin((pfx - fx) ** 2 + (pfy - fy) ** 2))


@pytest.mark.parametrize("want_fx, want_fy", [(0.5, 0.5), (0.25, 0.75), (0.8, 0.6), (0.15, 0.2)])
def test_a_sloped_unit_lands_on_the_terrain_pixels_actually_painted(want_fx, want_fy) -> None:
    """Track C6 supersedes this test's original premise for a 1x1 unit: it
    no longer sits at a sub-tile-dependent PIXEL RISE at all -- its marker
    paints through the exact same call as its own tile's terrain
    (render._sloped_tile_quad), so there is no separate placement to drift
    from the terrain's. want_fx/want_fy are kept as parameters (rather than
    dropped) precisely to pin that this is now true regardless of where on
    the tile the unit sits -- the old test's whole point, restated for the
    model that replaced it.
    """
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    ux, uy = 9, 8

    i = _diamond_pixel_nearest(tile_px, want_fx, want_fy)
    fp, fq = ig.tile_uv_fractions(tile_px)
    fx, fy = float(1 - fq[i]), float(fp[i])

    scn = _ramp_scenario(ux + fx, uy + fy)
    img, _elevations, corner_rise, proj = render.render_terrain_sloped_with_proj(scn)
    color = PLAYER_COLORS[_PLAYER % len(PLAYER_COLORS)]

    d_nw = int(corner_rise[uy, ux])
    d_ne = int(corner_rise[uy, ux + 1])
    d_sw = int(corner_rise[uy + 1, ux])
    d_se = int(corner_rise[uy + 1, ux + 1])
    d_min = min(d_nw, d_ne, d_sw, d_se)
    assert d_min > 0, "fixture sits at rise 0 -- the d_min convention could be dropped and still pass"
    assert len({d_nw, d_ne, d_sw, d_se}) > 1, "fixture tile is planar -- the (fx, fy) term proves nothing"

    # The unit's own tile's WHOLE warped footprint -- not a diamond, and not
    # scoped to (fx, fy) -- must now show the unit's colour, unconditionally.
    s_dst_y, s_dst_x, _sy, _sx, _uv = ig.sloped_quad_indices(tile_px, d_nw, d_ne, d_sw, d_se)
    base_x_t, base_y_t = ig.tile_screen_origin(ux, uy, 0, proj)
    rows, cols = base_y_t - d_min + s_dst_y, base_x_t + s_dst_x
    assert np.all(np.all(img[rows, cols] == color, axis=1)), (
        f"unit at sub-tile ({fx:.3f}, {fy:.3f}) did not paint every one of its own tile's "
        "warped terrain pixels its own colour -- the marker has drifted from the tile"
    )


def test_a_flat_map_places_sloped_units_exactly_where_stepped_does() -> None:
    """The strongest oracle Step 2 could have broken, in its unit-bearing
    form: on a flat map every corner is equal under any rule, so
    unit_rise_px must reduce to exactly elevation * elev_step -- integer, no
    rounding -- and render_terrain_sloped must stay byte-identical to
    render_terrain_iso with units included.

    Covered at the whole-image level by tests/test_sloped_render.py; kept
    here too, at a nonzero elevation and a fractional sub-tile position,
    because those are the two inputs C5 added and a flat map at elevation 0
    would pass with the rise term dropped entirely.
    """
    units = [[] for _ in range(9)]
    units[_PLAYER] = [SyntheticUnit(x=9.37, y=8.62, unit_const=_UNIT_CONST)]
    tiles = [SyntheticTile(x=x, y=y, elevation=3) for y in range(MAP_H) for x in range(MAP_W)]
    scn = FakeScenario(MAP_W, MAP_H, tiles, units)

    sloped, _elev, corner_rise, proj = render.render_terrain_sloped_with_proj(scn)
    stepped, _elev_i, _proj_i = render.render_terrain_iso_with_proj(scn)

    assert ig.unit_rise_px(corner_rise, 9, 8, 0.37, 0.62) == 3 * proj.elev_step
    assert sloped.shape == stepped.shape
    assert np.array_equal(sloped, stepped)


def test_the_bystander_headroom_covers_every_units_own_rise() -> None:
    """_unit_screen_bbox_iso's premise ("depends on proj + elevations,
    never on corner_rise") stops holding in Sloped once a unit sits at
    unit_rise_px. Under-covering there is a MISSED bystander paint, i.e. a
    stale pixel, not a no-op -- so this pins the bound itself rather than
    waiting for a chunk-seam artefact to show up.

    Non-vacuous by assertion: the fixture must contain at least one unit
    whose interpolated rise genuinely exceeds its own tile's
    elevation * elev_step, or a headroom of 0 would pass too.
    """
    units = [[] for _ in range(9)]
    units[_PLAYER] = [
        SyntheticUnit(x=x + 0.5, y=8.5, unit_const=_UNIT_CONST, reference_id=x)
        for x in range(4, 13)
    ]
    tiles = [
        SyntheticTile(x=x, y=y, elevation=max(0, min(4, x - 6)))
        for y in range(MAP_H)
        for x in range(MAP_W)
    ]
    scn = FakeScenario(MAP_W, MAP_H, tiles, units)

    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    headroom = render._unit_rise_headroom_px(corner_rise, elevations, proj)

    excesses = []
    for unit in scn.unit_manager.units[_PLAYER]:
        ux, uy = int(unit.x), int(unit.y)
        rise = ig.unit_rise_px(corner_rise, ux, uy, unit.x - ux, unit.y - uy)
        excesses.append(rise - int(elevations[uy, ux]) * proj.elev_step)

    assert max(excesses) > 0, "no unit sits above its own tile's level -- headroom=0 would pass too"
    assert headroom >= max(excesses), (
        f"headroom {headroom}px under-covers a unit sitting {max(excesses)}px above its own "
        "tile's level -- composite_rect_sloped would miss it as a bystander"
    )
