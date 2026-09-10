"""Verifies Sloped's real unit sprite path -- Track P3-g6.

Sloped previously drew every unit as a flat coloured diamond
(_draw_unit_sloped, no sprite branch at all). This module is the Sloped
counterpart to tests/test_sprite_chunks.py/tests/test_sprite_edit_bbox.py,
covering what's specific to Sloped rather than re-proving what those already
cover for Stepped:

  1. Flat-map byte identity WITH a real sprite-bearing unit -- the free
     oracle: on a flat map unit_rise_px() reduces exactly to
     `elevation * elev_step`, so
     the Sloped anchor expression must match Stepped's character for
     character. The blank template has no units of its own, so this
     deliberately places one -- an oracle a fixture can't exercise vacuously.
  2. Stitched chunks byte-identical to a full sprite render on a RAMPED
     fixture (a flat one exercises no resample code -- see
     tests/test_sloped_pick.py's own delegation-guard trap).
  3. Mark / sprite / pick agree on one unit's height on non-flat ground.
  4. Farms drape as real terrain in Sloped when sprites are on (Track C6,
     matching Stepped), and fall back to their plain mark when sprites are
     off (the recorded residual, still exercised via an explicit
     with_farms=False call).
  5. F2's one-ring dilation: an edit on a NEIGHBOUR of a sprite's own tile
     must still widen the dirty bbox to cover it, since a Sloped anchor
     reads its own tile's four corners rather than its centre tile alone.
  6. The Step 0 canvas-clip fix: a sprite anchored at the map's own
     worst-case corner must not lose its bottom edge at a low
     elev_step_pct stop, where Sloped's tight canvas (no skirt) would
     otherwise clip it more than Stepped ever does.

Synthetic sprite bytes on a tmp install throughout, matching this suite's
standing posture (test_sprite_chunks.py's own docstring)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, iso_geometry as ig, render, settings, terrain_palette, unit_pick, unit_sprites
from descape.render import (
    _dirty_screen_bbox,
    dirty_screen_bbox_sloped,
    render_terrain_iso_with_proj,
    render_terrain_sloped_with_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import SlopedChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit.fakes import FakeScenario, SyntheticTile
from test_unit_sprites import CONST, FILE_NAME, build_sld


@dataclass
class Unit:
    """Duck-typed, matching tests/test_sprite_edit_bbox.py's own Unit -- the
    four attributes _units_by_tile/sprite_draws_by_anchor/unit_tile_bounds
    actually read."""

    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """A tmp install holding one native-tile-sized synthetic sprite --
    copied from tests/test_sprite_toggle_viewer.py's fixture of the same
    name. Good enough for the flat-map oracle and the mark/sprite/pick
    agreement check, where only the ANCHOR position matters, not how far the
    sprite's own pixels reach -- tests that need real reach (F2, the Step 0
    canvas fix) use oversized_sprite_install instead."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                         "mirroring_mode": 6, "frame_count": 1}},
    )
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


SPRITE_CANVAS = 12 * unit_sprites.NATIVE_TILE_W
REACH_NAMES = ("MAX_SPRITE_REACH_LEFT", "MAX_SPRITE_REACH_UP", "MAX_SPRITE_REACH_RIGHT", "MAX_SPRITE_REACH_DOWN")


@pytest.fixture
def oversized_sprite_install(tmp_path, monkeypatch):
    """A sprite that reaches SPRITE_CANVAS // 2 == 576px past its hotspot on
    every side, with the four MAX_SPRITE_REACH_* constants pinned to match --
    copied from tests/test_sprite_edit_bbox.py's fixture of the same name,
    see its own docstring for why both the oversize AND the pin are
    required, not merely tidy. Needed wherever a test must out-reach a
    tile's own terrain-shaped bbox (test_sprite_edit_bbox.py's whole point):
    tests/test_sprite_toggle_viewer.py's native-tile-sized sprite is
    invisible next to a tile's own diamond+skirt sweep, so a fixture built
    on it can't isolate a widening bug from ordinary terrain coverage."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4, canvas=SPRITE_CANVAS))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                         "mirroring_mode": 6, "frame_count": 1}},
    )
    for name in REACH_NAMES:
        monkeypatch.setattr(unit_sprites, name, SPRITE_CANVAS // 2)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


# A const with a FOUNDATION_TERRAIN entry AND no .sld in the shipped graphic
# table -- _terrain_overlay_for() only resolves those (a real .sld always
# wins over a terrain override, and most FOUNDATION_TERRAIN entries are
# ordinary buildings that keep drawing as their sprite -- see that
# function's own docstring). 50 is Farm itself.
FARM_CONST = 50
assert FARM_CONST in terrain_palette.FOUNDATION_TERRAIN
assert FARM_CONST not in unit_sprites.graphic_map()


def _flat_scenario_with_unit(elevation: int, ux: float, uy: float, unit_const: int = CONST):
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = elevation
    scenario.unit_manager.units[1].append(Unit(ux, uy, unit_const))
    return scenario


def _ramped_scenario_with_unit(ux: float, uy: float, unit_const: int = CONST):
    """Climbs by one elevation level over the map's east half -- the same
    shape tests/test_sloped_chunks.py's own _load_sloped_scenario uses,
    plus one placed unit."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = 1 if tile.x >= mm.map_width // 2 else 0
    scenario.unit_manager.units[1].append(Unit(ux, uy, unit_const))
    return scenario


def _make_cache(scenario, **kwargs) -> SlopedChunkCache:
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    return SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, **kwargs)


def test_flat_map_byte_identical_with_sprites(sprite_install):
    """CONST here is never a farm const, so with_farms's default (True on
    both sides as of Track C6) never enters into it either way."""
    scenario = _flat_scenario_with_unit(elevation=2, ux=50.5, uy=50.5)
    sloped, _e, _c, sloped_proj = render_terrain_sloped_with_proj(scenario, with_sprites=True)
    stepped, _e2, stepped_proj = render_terrain_iso_with_proj(scenario, with_sprites=True)
    assert sloped_proj.canvas_w == stepped_proj.canvas_w
    assert sloped.shape == stepped.shape
    assert not np.array_equal(sloped, np.zeros_like(sloped)), "fixture painted nothing -- the sprite never resolved"
    assert np.array_equal(sloped, stepped)


def test_flat_map_byte_identical_with_a_farm(sprite_install):
    """Track C6's own flat-map byte-identity oracle: with with_farms now
    defaulting True on both sides, a FARM unit -- previously carved out of
    the check above precisely because Sloped forced with_farms=False -- must
    render identically too. On a flat map every corner of every tile is
    equal, so sloped_quad_indices/sloped_tile_edge_indices degenerate to
    diamond_indices/tile_edge_indices exactly (their own delegation
    contract), which is what makes this an EXACT match, not merely close."""
    scenario = _flat_scenario_with_unit(elevation=2, ux=50.5, uy=50.5, unit_const=FARM_CONST)
    sloped, _e, _c, sloped_proj = render_terrain_sloped_with_proj(scenario, with_sprites=True)
    stepped, _e2, stepped_proj = render_terrain_iso_with_proj(scenario, with_sprites=True)
    assert sloped_proj.canvas_w == stepped_proj.canvas_w
    assert sloped.shape == stepped.shape
    assert not np.array_equal(sloped, np.zeros_like(sloped)), "fixture painted nothing -- the farm never resolved"
    assert np.array_equal(sloped, stepped)


def test_stitched_chunks_match_full_render_on_a_ramp(sprite_install):
    scenario = _ramped_scenario_with_unit(ux=20.5, uy=20.5)
    cache = _make_cache(scenario, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims()
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
    full, _e, _c, _proj = render_terrain_sloped_with_proj(scenario, with_sprites=True)
    assert not np.array_equal(stitched, np.zeros_like(stitched)), "fixture painted nothing -- the sprite never resolved"
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


def test_mark_sprite_pick_agree_on_height(sprite_install):
    """One unit on a ramp: the plain-mark path's own rise_px (what
    _paint_tile_and_units_sloped computes inline for a mark), the sprite
    path's implied rise (recovered from sprite_draws_by_anchor's ay), and
    unit_pick.unit_rise_px_for must all report the identical pixel rise."""
    scenario = _ramped_scenario_with_unit(ux=60.3, uy=59.7, unit_const=CONST + 1)  # no sprite -> mark
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    unit = scenario.unit_manager.units[1][0]
    ux, uy = int(unit.x), int(unit.y)
    mark_rise = ig.unit_rise_px(corner_rise, ux, uy, unit.x - ux, unit.y - uy)

    entry = unit_pick.UnitEntry(player_id=1, unit=unit, own_x=ux, own_y=uy, order=0)
    pick_rise = unit_pick.unit_rise_px_for(entry, corner_rise)
    assert pick_rise == mark_rise

    # Now give the same unit a real sprite and recover the implied rise from
    # its own anchor y: ay = round(origin_y + (fy-fx)*half_h - rise_px) + half_h,
    # with fx/fy the FOOTPRINT CENTRE (own tile's centre for a span (1, 1)
    # unit -- _span_start(coord, 1) == int(coord)), not the unit's own exact
    # sub-tile position -- see sprite_draws_by_anchor's own fx/fy vs
    # own_fx/own_fy distinction.
    unit.unit_const = CONST
    sprites = render.sprite_draws_by_anchor(
        scenario, proj, elevations, corner_rise=corner_rise, with_farms=False
    )
    assert sprites.by_anchor, "fixture painted nothing -- the sprite never resolved"
    (_draw, _ax, ay), = next(iter(sprites.by_anchor.values()))
    half_h = proj.half_h
    fx, fy = ux + 0.5, uy + 0.5
    implied_rise = round(proj.origin_y + (fy - fx) * half_h) + half_h - ay
    assert implied_rise == mark_rise


def test_farms_drape_as_terrain_in_sloped():
    """Track C6's replacement for the deferral this test used to pin
    (test_farms_stay_marks_in_sloped, retired -- Sloped now has a
    warped-outline path, sprite_draws_by_anchor's with_farms default flipped
    to True at every Sloped call site, and this goes red if that ever
    reverts). With sprites on, a farm must resolve into farm_by_tile (so
    render._paint_tile_and_units_sloped's drape branch actually runs) and
    into skip_ids (so it never also falls through to the plain-diamond mark
    path)."""
    scenario = _flat_scenario_with_unit(elevation=0, ux=30.5, uy=30.5, unit_const=FARM_CONST)
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)

    sprites = render.sprite_draws_by_anchor(scenario, proj, elevations, corner_rise=corner_rise)
    assert sprites.farm_by_tile, "fixture is vacuous -- FARM_CONST never resolves a foundation terrain"
    assert sprites.skip_ids, "a draped farm must be skipped by the plain-mark path, not double-drawn"

    # The sprites-off residual: with_farms=False (still an explicit opt-out,
    # matching _paint_tile_and_units_sloped's own sprites-off fallback to
    # the plain mark) must never populate farm_by_tile.
    sprites_off = render.sprite_draws_by_anchor(
        scenario, proj, elevations, corner_rise=corner_rise, with_farms=False
    )
    assert sprites_off.farm_by_tile == {}, "with_farms=False must never populate farm_by_tile"
    assert not sprites_off.skip_ids, "a farm's unit must not be skipped when it has no sprite path to replace its mark"


def test_farm_perimeter_stroke_lands_on_the_warped_quad_not_the_elevation_based_spot(sprite_install):
    """Regression for the farm-drape plan's own named silent-failure trap:
    the perimeter stroke must use _render_tile_sloped's placement
    convention (elevation 0, corner_rise, -d_min folded into base_y), NOT
    _render_tile_iso's elevation-based one -- a copy-paste of the Stepped
    version would compile, run, and land the outline in the wrong place
    with no error.

    A one-tile-deep ISOLATED pit is the fixture, not an ordinary ramp: a
    farm corner tile sitting one level below its own uniformly-higher
    surroundings (legal under the +-1-neighbour invariant) is the one shape
    where `d_min` (the min of a tile's 4 corners, which every corner is
    pulled up towards by a taller neighbour under SLOPE_CORNER_RULE="max")
    provably exceeds that tile's OWN elevation -- an ordinary multi-tile
    ramp never produces this for any of a farm's OWN perimeter tiles, since
    a solid contiguous footprint always has at least one corner shared with
    another same-elevation tile in the block, which keeps d_min == the
    tile's own elevation there and made an earlier draft of this test pass
    under the WRONG placement by coincidence.

    Confirmed sensitive by hand against exactly this fixture: swapping the
    oracle's placement to `tile_screen_origin(tx, ty, elevations[ty, tx],
    proj)` (`_render_tile_iso`'s own elevation-based convention) shifts the
    checked pixels by 8px (one elev_step) and the assertion below goes red;
    an ordinary ramp fixture does NOT reproduce that, which is exactly why
    this fixture replaced one."""
    mm_w = mm_h = 20
    tiles = []
    for y in range(mm_h):
        for x in range(mm_w):
            tiles.append(SyntheticTile(x=x, y=y, elevation=0 if (x, y) == (9, 9) else 1))
    scenario = FakeScenario(mm_w, mm_h, tiles, [[]] * 1 + [[Unit(10.5, 10.5, FARM_CONST)]] + [[]] * 7)
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    tile_px = tile_pixels_for_map(mm_w, mm_h)
    tx, ty = 9, 9  # the pit tile itself: the farm's own NW corner tile

    sprites = render.sprite_draws_by_anchor(scenario, proj, elevations, corner_rise=corner_rise)
    assert (tx, ty) in sprites.farm_by_tile, "fixture painted nothing at the pit tile -- the farm never resolved there"
    _terrain_id, outline_color, edge_mask = sprites.farm_by_tile[(tx, ty)]
    assert edge_mask, "the pit tile must be on the farm's own perimeter (a nonzero edge_mask) or this proves nothing"

    d_nw = int(corner_rise[ty, tx])
    d_ne = int(corner_rise[ty, tx + 1])
    d_sw = int(corner_rise[ty + 1, tx])
    d_se = int(corner_rise[ty + 1, tx + 1])
    own_elev_px = int(elevations[ty, tx]) * proj.elev_step
    assert min(d_nw, d_ne, d_sw, d_se) > own_elev_px, (
        "fixture is not a genuine isolated pit -- d_min must exceed the tile's own elevation, "
        "or the elevation-based mutation would coincidentally land in the same place"
    )

    img, _e, _c, _proj = render_terrain_sloped_with_proj(scenario, with_sprites=True)
    base_x, base_y, _dy, _dx, _sy, _sx, _uv = render._sloped_tile_quad(
        tx, ty, tile_px, proj, d_nw, d_ne, d_sw, d_se
    )

    checked = 0
    for bit, side in render._FARM_EDGE_BITS:
        if not (edge_mask & bit):
            continue
        edge_y, edge_x = ig.sloped_tile_edge_indices(tile_px, side, d_nw, d_ne, d_sw, d_se)
        rows, cols = base_y + edge_y, base_x + edge_x
        assert np.all(np.all(img[rows, cols] == outline_color, axis=1)), (
            f"tile ({tx}, {ty}) side={side} does not show the outline colour at its own "
            "warped placement -- the stroke landed somewhere else"
        )
        checked += 1
    assert checked > 0, "the pit tile's edge_mask had no set bits -- the check would be vacuous"


def test_dirty_bbox_widens_for_a_neighbour_edit(oversized_sprite_install):
    """F2: a Sloped anchor reads its own tile's four CORNERS, each shared
    with up to four tiles, so editing a NEIGHBOUR of the sprite's own tile
    (never the tile itself) must still widen the dirty bbox enough to
    repaint it. sprite_band_radius=0 (Stepped's own exact shape) must NOT
    cover it -- that is the mutation this test is designed to catch.

    with_units=False here is a TEST ISOLATION choice, not a claim about a
    real call site (_apply_dirty's Sloped branch always passes with_units=
    True today): with it on, the general seed's own UNIT_FOOTPRINT_MAX_
    RADIUS=4 dilation already sweeps a tile block wide enough to
    accidentally overlap even an oversized sprite by coincidence, which
    would make radius=0 pass for the wrong reason. False shrinks the
    general seed to a plain 1-tile ring, isolating what only
    sprite_band_radius adds -- calling the internal _dirty_screen_bbox
    directly, as this test does, can exercise combinations no production
    caller uses yet."""
    scenario = _ramped_scenario_with_unit(ux=61.5, uy=60.5)
    mm = scenario.map_manager
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    sprites = render.sprite_draws_by_anchor(
        scenario, proj, elevations, corner_rise=corner_rise, with_farms=False
    )
    assert sprites.bboxes, "fixture painted nothing -- the sprite never resolved"
    sx0, sy0, sx1, sy1 = next(iter(sprites.bboxes.values()))

    # A tile diagonally adjacent to the unit's own (61, 60) tile, not the
    # tile itself, so this is genuinely "F2, not the ordinary seed dilation".
    dirty_indices = [i for i, t in enumerate(mm.terrain) if (t.x, t.y) == (62, 61)]
    assert dirty_indices, "test setup: neighbour tile not found"

    canvas_dims = render._canvas_pixel_dims(proj)
    exact = _dirty_screen_bbox(
        scenario, dirty_indices, elevations.copy(), proj, canvas_dims, with_units=False,
        with_sprites=True, sprite_band_radius=0,
    )
    dilated = _dirty_screen_bbox(
        scenario, dirty_indices, elevations.copy(), proj, canvas_dims, with_units=False,
        with_sprites=True, sprite_band_radius=1,
    )

    def _contains(bbox, sx0, sy0, sx1, sy1):
        bx0, by0, bx1, by1 = bbox
        return bx0 <= sx0 and by0 <= sy0 and bx1 >= sx1 and by1 >= sy1

    assert not _contains(exact, sx0, sy0, sx1, sy1), (
        "radius=0 (Stepped's own shape) already covers the sprite -- fixture doesn't "
        "isolate F2, pick a neighbour tile further from the sprite's own tile"
    )
    assert _contains(dilated, sx0, sy0, sx1, sy1), "F2's one-ring dilation failed to widen far enough"

    # dirty_screen_bbox_sloped's own with_sprites=True call must use the
    # dilated shape, not the exact one -- the actual production wiring.
    wired = dirty_screen_bbox_sloped(
        scenario, dirty_indices, elevations.copy(), proj, with_units=True, with_sprites=True,
    )
    assert _contains(wired, sx0, sy0, sx1, sy1)


def test_south_edge_sprite_not_clipped_at_low_elev_step_pct(tmp_path, monkeypatch):
    """Step 0's canvas-clip fix: at a low elev_step_pct stop, Sloped's tight
    canvas_dims() (no skirt, unlike Stepped) clips a sprite's bottom edge
    more than Stepped's own accepted gap does. Placed at (0, h-1) -- the
    tile canvas_size_and_origin's own derivation names as the one with the
    LEAST negative (tallest final) screen position, i.e. this map's own
    worst case for a sprite reaching further down.

    Neither of this module's other two sprite_install fixtures fits here:
    the native-tile-sized one's own downward reach doesn't reliably clear
    the ~60px gap (tile_px=64 at this pct stop), and oversized_sprite_
    install's 576px reach blows straight past it in BOTH the fixed and
    unfixed canvas -- measured, not assumed, when this test was first
    written against it. This fixture is sized in between, computed once
    (not re-derived per run) so the sprite's bbox bottom edge lands inside
    the fix's own [tight_h, padded_h) window rather than short of or past
    it -- see the SPRITE_CANVAS_FOR_CLIP_TEST comment for the numbers.

    This is a positive check that the fix is active, not a synthetic
    mutation: canvas_dims() must be wider than the bare proj bound once
    sprites are enabled, and the sprite's own bottom row must actually
    receive non-background pixels once composited into that wider bound."""
    monkeypatch.setattr(settings, "_elev_step_pct", 25)
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = 0
    ux, uy = 0.5, mm.map_height - 0.5
    scenario.unit_manager.units[1].append(Unit(ux, uy, CONST))

    # At this fixture's elevation/placement/tile_px (measured, see the
    # module docstring above): tight_h=3900, padded_h=3960 (a 60px gap).
    # canvas=200 (native reach 100 on every side) puts the sprite's own
    # bbox bottom at y1=3950 -- inside the gap, clear of both edges.
    #
    # playercolor=False: build_sld's default PLAYERCOLOR mask renders as
    # solid black on this bare synthetic install (no real player-colour
    # palette configured), indistinguishable from this array's own
    # zero-initialized background -- measured while first writing this test,
    # not assumed. The MAIN layer alone decodes non-black.
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4, canvas=200, playercolor=False))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                         "mirroring_mode": 6, "frame_count": 1}},
    )
    for name in REACH_NAMES:
        monkeypatch.setattr(unit_sprites, name, 100)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    try:
        cache = _make_cache(scenario, sprites=True)
        tight_h = cache.proj.canvas_h
        canvas_w, canvas_h = cache.canvas_dims()
        assert canvas_h > tight_h, "canvas_dims() must widen once sprites are enabled at a low pct stop"

        stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
        assert not np.array_equal(stitched, np.zeros_like(stitched)), "fixture painted nothing at all"
        bottom_strip = stitched[tight_h:canvas_h]
        assert bottom_strip.any(), (
            "the newly-widened strip below the tight canvas bound is still all-background -- "
            "the sprite's own bottom edge is being clipped exactly where the fix should prevent it"
        )
    finally:
        asset_source.set_install_path_override(None)
        unit_sprites.clear_caches()
