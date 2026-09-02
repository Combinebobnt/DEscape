"""Multi-tile buildings must fill their WHOLE footprint in Stepped and Sloped.

The defect this covers: _units_by_tile() bucketed a unit on its own floored
tile only, and _draw_unit_iso() painted the unit's entire footprint at that
one moment in the depth walk -- so every footprint tile later in depth_order
painted its own terrain diamond back over the slab. A House kept 75% of its
footprint, a Colosseum 51.6%.

numpy only, no Qt: render.render_terrain_iso_with_proj() returns
(img, elevations, proj) in one call, so the whole check is a crop-and-compare
against tile_screen_origin(). The synthetic idiom (SyntheticTile /
SyntheticUnit / FakeScenario / _flat_scenario, and _BUILDING_UNIT_CONST
derived by span lookup rather than hardcoded) is the established one from
tools/verify_iso_units.py.
"""

from __future__ import annotations


import numpy as np

from testkit.fakes import (
    FakeScenario,
    SyntheticTile,
    SyntheticUnit,
)

from descape import iso_geometry, render
from descape.terrain_palette import BUILDING_TILE_SPANS, PLAYER_COLORS




# The SPAN is fixed; only the const answering to it is looked up, so this
# doesn't silently stop testing anything if the .dat's data changes shape.
_BUILDING_SPAN = 4
_BUILDING_UNIT_CONST = next(
    uid for uid, (sx, sy) in BUILDING_TILE_SPANS.items() if sx == _BUILDING_SPAN and sy == _BUILDING_SPAN
)

MAP_W = MAP_H = 16
_PLAYER = 1


def _flat_scenario(w: int, h: int, units_by_player) -> FakeScenario:
    tiles = [SyntheticTile(x=x, y=y, elevation=0) for y in range(h) for x in range(w)]
    return FakeScenario(w, h, tiles, units_by_player)


def _building_scenario(bx: float, by: float, raised: tuple[int, int] | None = None) -> FakeScenario:
    units = [[] for _ in range(9)]
    units[_PLAYER] = [SyntheticUnit(x=bx, y=by, unit_const=_BUILDING_UNIT_CONST)]
    scn = _flat_scenario(MAP_W, MAP_H, units)
    if raised is not None:
        scn.map_manager.get_tile(*raised).elevation = 1
    return scn


def _unit_color() -> tuple[int, int, int]:
    return PLAYER_COLORS[_PLAYER % len(PLAYER_COLORS)]


def _diamond_block(img: np.ndarray, tx: int, ty: int, elevation: int, proj, tile_px: int) -> np.ndarray:
    """The pixels of one tile's diamond, as an (N, 3) array."""
    dst_y, dst_x, _sy, _sx = iso_geometry.diamond_indices(tile_px)
    bx, by = iso_geometry.tile_screen_origin(tx, ty, elevation, proj)
    return img[by + dst_y, bx + dst_x]


def test_a_multi_tile_building_fills_every_footprint_tile_on_flat_ground() -> None:
    """The assertion that fails today, and the whole point of the fix.

    A flat map has no skirts and no contact shadow over the footprint, so
    exact colour equality is sound here and 100% per tile is the honest bar
    -- not a tolerance chosen to make a partial fill pass.
    """
    scn = _building_scenario(10.5, 10.5)
    img, elevations, proj = render.render_terrain_iso_with_proj(scn)
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    color = _unit_color()

    unit = scn.unit_manager.units[_PLAYER][0]
    x0, x1, y0, y1 = render.unit_tile_bounds(unit, MAP_W, MAP_H)
    assert (x1 - x0, y1 - y0) == (_BUILDING_SPAN, _BUILDING_SPAN)
    own_elev = int(elevations[int(unit.y), int(unit.x)])

    shortfalls = []
    for ty in range(y0, y1):
        for tx in range(x0, x1):
            block = _diamond_block(img, tx, ty, own_elev, proj, tile_px)
            matched = int(np.count_nonzero(np.all(block == color, axis=1)))
            if matched != block.shape[0]:
                shortfalls.append(((tx, ty), matched, block.shape[0]))

    total = (x1 - x0) * (y1 - y0)
    assert not shortfalls, (
        f"{len(shortfalls)}/{total} footprint tiles are not solidly the unit colour: "
        f"{shortfalls[:6]}{' ...' if len(shortfalls) > 6 else ''}"
    )


def test_a_multi_tile_building_fills_its_footprint_on_sloped_ground_as_one_flat_pad() -> None:
    """Sloped's half of the fix, and the guard on its one fragile edit.

    `_paint_tile_and_units_sloped` calls `iso_geometry.unit_rise_px` PER
    ENTRY, over the UNIT'S OWN tile's four corners. Hoisting that call back
    out of the loop looks like a textbook loop-invariant cleanup, and is
    wrong: under per-footprint-tile bucketing `tile` is usually a footprint
    tile, not the unit's own one, so the hoisted version derives each
    diamond's height from a different tile and the slab conforms to the
    slope instead of staying one flat pad.

    Track C5's Step 2 changed the height EXPRESSION here (a 4-corner
    average quantized to an integer elevation level became the exact
    interpolated pixel rise of the surface the compositor paints); the
    containment property below is unchanged by that, which is the point of
    guarding it separately from the expression.

    The fixture is a ramp, so corner_rise genuinely differs across the
    footprint -- asserted below, because on uniform ground both versions
    agree and the test would prove nothing.

    The assertion is containment, not coverage: **no unit-coloured pixel may
    fall outside the union of the 16 diamonds at the own-tile rise.** That
    is the occlusion-proof direction. Higher terrain legitimately buries
    part of a flat pad on a ramp, and occlusion only ever REMOVES unit
    pixels from that union -- it can never put one outside it. The hoisted
    version, placing diamonds at each footprint tile's own height, paints
    outside the union immediately. Liveness is guarded separately so a
    render that drew nothing at all can't pass containment vacuously.
    """
    units = [[] for _ in range(9)]
    units[_PLAYER] = [SyntheticUnit(x=10.5, y=10.5, unit_const=_BUILDING_UNIT_CONST)]
    tiles = [
        SyntheticTile(x=x, y=y, elevation=max(0, min(4, x - 8)))
        for y in range(MAP_H)
        for x in range(MAP_W)
    ]
    scn = FakeScenario(MAP_W, MAP_H, tiles, units)

    img, _elev, corner_rise, proj = render.render_terrain_sloped_with_proj(scn)
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    color = _unit_color()

    unit = scn.unit_manager.units[_PLAYER][0]
    x0, x1, y0, y1 = render.unit_tile_bounds(unit, MAP_W, MAP_H)
    ux, uy = int(unit.x), int(unit.y)

    # Non-vacuity: the footprint must actually straddle a height change.
    sub = corner_rise[y0 : y1 + 1, x0 : x1 + 1]
    spread = int(sub.max() - sub.min())
    assert spread > 0, "fixture went flat across the footprint -- the hoist trap would pass too"

    # The one rise every diamond must sit at, derived exactly as the
    # compositor does it: the unit's OWN tile's corners, at its OWN
    # sub-tile position.
    rise = iso_geometry.unit_rise_px(corner_rise, ux, uy, unit.x - ux, unit.y - uy)

    dst_y, dst_x, _sy, _sx = iso_geometry.diamond_indices(tile_px)
    allowed = np.zeros(img.shape[:2], dtype=bool)
    live = 0
    for ty in range(y0, y1):
        for tx in range(x0, x1):
            bx, by = iso_geometry.tile_screen_origin(tx, ty, 0, proj)
            by -= rise
            allowed[by + dst_y, bx + dst_x] = True
            block = img[by + dst_y, bx + dst_x]
            if int(np.count_nonzero(np.all(block == color, axis=1))):
                live += 1

    is_unit = np.all(img == color, axis=2)
    stray = int(np.count_nonzero(is_unit & ~allowed))
    assert stray == 0, (
        f"{stray} unit-coloured pixels fall outside the 16 diamonds at the unit's own-tile "
        f"rise {rise}px -- the slab is conforming to the slope instead of staying a "
        "flat pad (the per-entry unit_rise_px call was hoisted out of the loop?)"
    )
    assert live == (x1 - x0) * (y1 - y0), (
        f"only {live}/{(x1 - x0) * (y1 - y0)} footprint tiles carry any unit colour at all -- "
        "containment above would pass vacuously"
    )


def _render_without_shading(scn) -> np.ndarray:
    """render_terrain_iso() with BOTH darkening passes neutralized to 1.0.

    Raising a tile darkens its up-screen neighbours through two independent
    lru_cached factor tables -- _shadow_factors (the contact band and its
    apex) and _seam_factors (the seam line and its apex). Neutralizing only
    the first, as tools/verify_iso_units.py's older idiom does, leaves the
    seam free to change pixels for a reason that has nothing to do with
    occlusion. Both are set to identity (the code paths still run, they just
    multiply by 1), and both caches are cleared on the way in AND in the
    finally -- they are lru_caches, so a stale factor array would leak into
    every later test in the session.
    """
    original_contact, original_seam = render.CONTACT_SHADE, render.SEAM_SHADE
    render.CONTACT_SHADE = 1.0
    render.SEAM_SHADE = 1.0
    render._shadow_factors.cache_clear()
    render._seam_factors.cache_clear()
    try:
        return render.render_terrain_iso(scn)
    finally:
        render.CONTACT_SHADE = original_contact
        render.SEAM_SHADE = original_seam
        render._shadow_factors.cache_clear()
        render._seam_factors.cache_clear()


def test_a_raised_tile_occludes_only_the_further_half_of_a_buildings_footprint() -> None:
    """Terrain-vs-unit occlusion WITHIN one building's footprint.

    tools/verify_iso_units.py's check_occlusion deliberately uses a
    single-tile unit (its shadow-neutralisation control depends on one known
    back-neighbour relationship), so the multi-tile property lives here.

    The geometry gives both directions at once. The building's own tile
    (10, 10) has d = y - x = 0; the raised tile (11, 10) has d = -1, so it
    paints AFTER every footprint tile with d < -1 and BEFORE every footprint
    tile with d > -1. Its two direct up-screen neighbours (12, 10) and
    (11, 9) are both footprint tiles, and both must lose unit pixels; every
    footprint tile closer than the raised one must be byte-identical.

    This is a real regression test, not a restatement: with the unit's whole
    slab painted at its own tile's moment, the raised tile paints before all
    of it and occludes nothing at all.
    """
    flat_img = _render_without_shading(_building_scenario(10.5, 10.5))
    raised_img = _render_without_shading(_building_scenario(10.5, 10.5, raised=(11, 10)))
    assert flat_img.shape == raised_img.shape

    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    color = _unit_color()
    proj = iso_geometry.canvas_size_and_origin(
        MAP_W, MAP_H, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
    )
    rx, ry = 11, 10
    d_raised = ry - rx

    def unit_px(img, tx, ty) -> int:
        block = _diamond_block(img, tx, ty, 0, proj, tile_px)
        return int(np.count_nonzero(np.all(block == color, axis=1)))

    # Direction 1: the raised tile's own up-screen neighbours are painted
    # earlier, so its diamond covers part of theirs.
    for tx, ty in ((rx + 1, ry), (rx, ry - 1)):
        assert ty - tx < d_raised, "fixture bug: that neighbour is not further than the raised tile"
        before, after = unit_px(flat_img, tx, ty), unit_px(raised_img, tx, ty)
        assert before > 0, f"footprint tile {(tx, ty)} had no unit pixels to occlude"
        assert after < before, (
            f"footprint tile {(tx, ty)} kept all {before} unit pixels after raising {(rx, ry)} -- "
            "a further footprint tile was not occluded"
        )

    # Direction 2: everything closer than the raised tile paints after it,
    # and must be untouched down to the byte.
    x0, x1, y0, y1 = render.unit_tile_bounds(
        _building_scenario(10.5, 10.5).unit_manager.units[_PLAYER][0], MAP_W, MAP_H
    )
    checked = 0
    for ty in range(y0, y1):
        for tx in range(x0, x1):
            if ty - tx <= d_raised:
                continue
            before = _diamond_block(flat_img, tx, ty, 0, proj, tile_px)
            after = _diamond_block(raised_img, tx, ty, 0, proj, tile_px)
            assert np.array_equal(before, after), (
                f"footprint tile {(tx, ty)} (d={ty - tx}) is closer than the raised tile "
                f"(d={d_raised}) and must not have changed"
            )
            checked += 1
    assert checked >= 8, f"only {checked} closer footprint tiles checked -- the fixture went vacuous"
