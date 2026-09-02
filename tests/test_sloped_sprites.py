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
  4. Farms keep their plain mark in Sloped even with sprites on
     (with_farms=False's suppression, not a paint-time skip).
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
    """Only exact for a map with no FARM units: Sloped's own call always
    passes with_farms=False (the deferral), Stepped's default is True, so a
    farm unit would resolve differently on each side and this comparison
    would fail for a reason unrelated to the anchor-height rewrite this test
    exists to catch. CONST here is never a farm const, so this fixture is
    unaffected -- see test_farms_stay_marks_in_sloped for that behaviour."""
    scenario = _flat_scenario_with_unit(elevation=2, ux=50.5, uy=50.5)
    sloped, _e, _c, sloped_proj = render_terrain_sloped_with_proj(scenario, with_sprites=True)
    stepped, _e2, stepped_proj = render_terrain_iso_with_proj(scenario, with_sprites=True)
    assert sloped_proj.canvas_w == stepped_proj.canvas_w
    assert sloped.shape == stepped.shape
    assert not np.array_equal(sloped, np.zeros_like(sloped)), "fixture painted nothing -- the sprite never resolved"
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


def test_farms_stay_marks_in_sloped():
    """Sloped has no warped-outline path for a farm foundation yet
    (_render_tile_sloped has no terrain_override at all), so
    sprite_draws_by_anchor() must be called with with_farms=False here --
    a paint-time skip alone would make a farm invisible instead of
    deferred, since _paint_tile_and_units_sloped's skip_ids gate has no
    farm-terrain-override branch to fall into (unlike Stepped's
    _paint_tile_and_units_iso)."""
    scenario = _flat_scenario_with_unit(elevation=0, ux=30.5, uy=30.5, unit_const=FARM_CONST)
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)

    # with_farms=True (Stepped's own default) DOES resolve this fixture as a
    # farm -- proves the fixture itself is non-vacuous before trusting the
    # False arm below.
    with_farms_true = render.sprite_draws_by_anchor(scenario, proj, elevations, corner_rise=corner_rise)
    assert with_farms_true.farm_by_tile, "fixture is vacuous -- FARM_CONST never resolves a foundation terrain"

    sprites = render.sprite_draws_by_anchor(
        scenario, proj, elevations, corner_rise=corner_rise, with_farms=False
    )
    assert sprites.farm_by_tile == {}, "with_farms=False must never populate farm_by_tile"
    assert not sprites.skip_ids, "a farm's unit must not be skipped when it has no sprite path to replace its mark"


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
