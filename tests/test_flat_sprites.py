"""Flat mode's footprint-fitted unit icons (P3-g7).

Flat is a top-down square grid, not an isometric one -- render_terrain paints
axis-aligned tile_px squares, not diamonds -- so a 2:1 iso sprite has no
natural placement there. The answer is a contain-fitted ICON centred in the
unit's own footprint rect, and the property everything below exists to pin is
that **an icon never leaves that rect**. That single property is what lets
Flat keep its existing per-tile edit rects and its exact canvas sizing, with
none of the dirty-bbox widening (`merge_sprite_bboxes`), canvas headroom or
`MAX_SPRITE_REACH_*` machinery Stepped's P3-g3 and Sloped's P3-g6 both needed.
Break it and every one of those is silently missing rather than deliberately
absent.

**The fixture here is its own synthetic unit-bearing scenario**, in
tests/test_sprite_chunks.py's `_Scenario` duck-type shape, deliberately NOT
tests/test_flat_chunks.py's default-tier one -- that is a unit-free blank
template (see its own note), so anything built on it would run vacuous. Every
check below asserts non-vacuity before asserting its real property.

Synthetic bytes and a tmp install throughout -- no game assets, matching the
suite's standing posture.

**Mutation arms, run by hand-reverting on 2026-09-03 rather than asserted**, so
what each check actually covers is recorded rather than hoped for:

- Centre an icon on its intersection with the requested rect instead of on its
  whole footprint: RED (the stitched-chunk check, and the footprint check at
  tile_px=64).
- Drop contain-fit's vertical constraint (`scale = fw / ink_w`): RED, via the
  aspect check below -- and NOT via the footprint check, because
  unit_sprites._build_icon's `min(fw/fh, ...)` clamp catches the overflow. That
  is the one mutation under which that clamp is live.
- Skip a unit in `_flat_icon_layer` WITHOUT advancing its row counter (the
  desync that stays in range and shifts every later icon onto its neighbour):
  RED, 9 of 11, via `FlatChunkCache._level_icons`' row-count assert.
- Drop that clamp on its own, keeping the correct scale: GREEN, and reported as
  green. `ink_w * scale <= fw` already holds by construction and float error is
  nowhere near the 0.5 `round()` would need to cross the boundary, so the clamp
  is an unreachable guard rather than a pinned behaviour. Kept anyway, for the
  case the mutation above demonstrates.
- Revert `icon_for()` to the iso facing zero point (drop its
  `FLAT_ANGLE_ZERO_OFFSET_DEG` arguments and let the default stand): RED, 4 of
  4, via the facing check at the bottom of this file. Recorded 2026-09-08, and
  it is the arm that matters for that change: every constant-level property in
  tests/test_unit_sprites.py stays green under it, because they exercise
  `angle_index`/`_frame_for` directly and never ask whether anything calls them
  with the flat value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, iso_geometry, render, render_cache, unit_sprites
from descape.terrain_palette import PLAYER_COLORS
from descape.unit_filter import UnitFilter
from test_unit_sprites import CONST, FILE_NAME, build_sld

MAP_W = MAP_H = 12
CHUNK = 128

# A second const, given a real 3x3 span below, so at least one footprint
# straddles a chunk seam at every mip level -- the case that catches an icon
# centred on its clipped rect instead of on its whole footprint.
BIG_CONST = CONST + 1
BIG_SPAN = (3, 3)

# A third const for the facing-offset wiring check, and its angle_count is the
# whole point of it being separate. The two above store 4 angles, where 45
# degrees is half a step, so angle_index() discards the iso offset and Flat and
# Stepped resolve the SAME frame. A wiring check on either would be green
# with icon_for() still passing the iso default. 16 is where they diverge, and
# is also the angle_count that dominates the real install (1,163 graphics).
# Non-composite on purpose: _assembled_native() then returns the single piece
# as-is, so the icon's pixels are that one frame's, uncomposited.
ANGLE_CONST = CONST + 2
ANGLE_FILE = "t_facing_x1"
ANGLE_COUNT = 16


@dataclass
class Tile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0
    layer: int = -1


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


class _MapManager:
    def __init__(self, tiles):
        self.map_width, self.map_height = MAP_W, MAP_H
        self.terrain = tiles
        self._by_xy = {(t.x, t.y): t for t in tiles}

    def get_tile(self, x, y):
        return self._by_xy[(x, y)]


class _UnitManager:
    def __init__(self, units_by_player):
        self.units = units_by_player


class _Scenario:
    def __init__(self, tiles, units_by_player):
        self.map_manager = _MapManager(tiles)
        self.unit_manager = _UnitManager(units_by_player)
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))


def _scenario(units_by_player):
    tiles = [Tile(x, y, 0) for y in range(MAP_H) for x in range(MAP_W)]
    return _Scenario(tiles, units_by_player)


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """A tmp install holding one graphic, registered as a two-piece composite
    so its assembled ink is deliberately NON-SQUARE (96 wide by 192 tall).

    Squareness is the thing to avoid, not a detail: contain-fitting a square
    ink into a square footprint fills it exactly, which makes "the icon stays
    inside its footprint" true for the wrong reason and hides both the
    centring and the aspect rule. A 1:2 ink into a square cell is half-width
    and full-height, which is checkable in both axes at once.
    """
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(
        build_sld(4, canvas=unit_sprites.NATIVE_TILE_W)
    )
    # build_sld() varies each frame's colour by its own index, so which angle a
    # rotation selects is readable straight off the icon's pixels.
    (graphics / f"{ANGLE_FILE}.sld").write_bytes(
        build_sld(ANGLE_COUNT, canvas=unit_sprites.NATIVE_TILE_W)
    )

    def entry(const):
        piece = {"unit_id": const, "file_name": FILE_NAME, "angle_count": 4,
                 "frame_count": 1, "dx": 0, "dy": 0}
        below = dict(piece, unit_id=const + 100, dy=unit_sprites.NATIVE_TILE_W)
        return {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                "mirroring_mode": 6, "frame_count": 1, "pieces": [piece, below]}

    facing = {"graphic_id": 2, "file_name": ANGLE_FILE, "angle_count": ANGLE_COUNT,
              "mirroring_mode": 6, "frame_count": 1}
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: entry(CONST), BIG_CONST: entry(BIG_CONST), ANGLE_CONST: facing},
    )
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, BIG_CONST, BIG_SPAN)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _units():
    """Two players' worth, mixing 1x1 units with a 3x3 building whose
    footprint crosses a chunk seam (tiles 1..3 span pixels 64..256 at
    tile_px=64, and CHUNK is 128)."""
    return [
        [Unit(9.0, 2.0, CONST)],
        [Unit(2.0, 2.0, BIG_CONST), Unit(6.0, 5.0, CONST), Unit(7.0, 9.0, CONST)],
    ]


def _mip_levels():
    return sorted(iso_geometry.mip_tile_px_candidates(
        render.tile_pixels_for_map(MAP_W, MAP_H)
    ).values())


def _full(scn, *, with_sprites):
    """A full Flat render the chunk path is compared against -- deliberately
    render_terrain + overlay_units, the INDEPENDENT implementation, never
    composite_rect_flat. See composite_rect_flat()'s own docstring on why
    sharing one would make the comparison a tautology."""
    return render.overlay_units(render.render_terrain(scn), scn, with_sprites=with_sprites)


def _stitched(cache, mip=0):
    w, h = cache.canvas_dims(mip)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y0 in range(0, h, CHUNK):
        for x0 in range(0, w, CHUNK):
            x1, y1 = min(x0 + CHUNK, w), min(y0 + CHUNK, h)
            out[y0:y1, x0:x1] = cache._composite_rect(mip, x0, y0, x1, y1)
    return out


def _footprint_rects(scn, tile_px):
    mm = scn.map_manager
    rects = []
    for units in scn.unit_manager.units:
        for unit in units:
            bounds = render.unit_tile_bounds(unit, mm.map_width, mm.map_height)
            if bounds is None:
                continue
            tx0, tx1, ty0, ty1 = bounds
            rects.append((tx0 * tile_px, ty0 * tile_px, tx1 * tile_px, ty1 * tile_px))
    return rects


# --- the icon never leaves its footprint ------------------------------


@pytest.mark.parametrize("tile_px", _mip_levels())
def test_an_icon_paints_only_inside_its_own_footprint(sprite_install, tile_px):
    """**The assertion P3-g7's whole no-widening argument rests on**, checked
    at every shipped mip level including the smallest (16), where the round-to-
    integer margins are proportionally largest. Every pixel that sprites-on
    changes relative to bare terrain must lie inside some unit's footprint
    rect -- one pixel outside and Flat needs the dirty-bbox widening and canvas
    headroom this whole design exists to avoid."""
    scn = _scenario(_units())
    mip = {px: level for level, px in
           iso_geometry.mip_tile_px_candidates(render.tile_pixels_for_map(MAP_W, MAP_H)).items()}[tile_px]

    terrain_only = _stitched(render_cache.FlatChunkCache(scn, render.tile_pixels_for_map(MAP_W, MAP_H),
                                                         with_units=False), mip)
    with_icons = _stitched(render_cache.FlatChunkCache(scn, render.tile_pixels_for_map(MAP_W, MAP_H),
                                                       sprites=True), mip)

    changed = np.any(terrain_only != with_icons, axis=2)
    assert changed.any(), "no unit painted anything, so this proves nothing"

    inside = np.zeros_like(changed)
    for x0, y0, x1, y1 in _footprint_rects(scn, tile_px):
        inside[y0:y1, x0:x1] = True
    stray = np.argwhere(changed & ~inside)
    assert len(stray) == 0, f"{len(stray)} px painted outside every footprint, first at {stray[:3].tolist()}"


def test_an_icon_contain_fits_and_leaves_the_footprint_partly_bare(sprite_install):
    """The render-level half of the contain-fit decision: a 1:2 ink in a square
    cell must leave bare terrain down both sides. Stretch-to-fill (the rejected
    alternative) would repaint the whole rect, and so would resolving only the
    composite's parent piece -- which would give a square 1:1 ink."""
    scn = _scenario([[], [Unit(6.0, 5.0, CONST)]])
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    # Bare terrain, NOT the coloured-mark render: a mark fills the whole cell,
    # so diffing against it would report the whole cell as changed and say
    # nothing about where the icon's own ink landed.
    terrain_only = render.render_terrain(scn)
    with_icons = _full(scn, with_sprites=True)

    x0, y0, x1, y1 = _footprint_rects(scn, tile_px)[0]
    changed = np.any(terrain_only[y0:y1, x0:x1] != with_icons[y0:y1, x0:x1], axis=2)
    assert changed.any(), "the icon resolved to nothing, so this proves nothing"
    painted_cols = np.flatnonzero(changed.any(axis=0))
    assert painted_cols[0] > 0 and painted_cols[-1] < (x1 - x0 - 1), (
        "the icon filled its cell edge to edge -- that is stretch-to-fill, not contain-fit"
    )
    assert changed.any(axis=1).all(), "a 1:2 ink should reach the full height of a square cell"


# --- the chunk path agrees with the independent full render -----------


def test_stitched_chunks_match_the_full_sprite_render(sprite_install):
    """Two properties at once, because for Flat they are the same assertion:
    chunks composited in isolation stitch back to the full render, AND
    overlay_units(with_sprites=True) -- the independent second implementation
    -- agrees with composite_rect_flat's icon branch pixel for pixel.

    The 3x3 building's footprint deliberately straddles a chunk seam, which is
    what catches an icon centred on its intersection with the requested rect
    rather than on its whole footprint: that bug is invisible in any single
    chunk and in any full-canvas render.
    """
    scn = _scenario(_units())
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    cache = render_cache.FlatChunkCache(scn, tile_px, sprites=True)
    assert cache._level_icons(0), "no unit resolved to an icon, so this proves nothing"

    seams = {x for x, _y, x1, _y1 in _footprint_rects(scn, tile_px) for x in (x, x1)}
    assert any(0 < s % CHUNK for s in seams if s % CHUNK), "no footprint crosses a chunk seam"

    assert np.array_equal(_stitched(cache), _full(scn, with_sprites=True))


def test_sprites_off_is_byte_identical_to_the_pre_sprite_render(sprite_install):
    """At every mip level. With icons=None composite_rect_flat's painted code
    path is unchanged bytes, so this is structural rather than a coincidence
    -- but the wiring above it (a cache built with sprites=False still asking
    for a layer, say) is not, and that is what this catches."""
    scn = _scenario(_units())
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    marks = _full(scn, with_sprites=False)
    cache = render_cache.FlatChunkCache(scn, tile_px, sprites=False)

    assert cache._level_icons(0) is None
    assert np.array_equal(_stitched(cache), marks)
    for level in iso_geometry.mip_tile_px_candidates(tile_px):
        assert cache._level_icons(level) is None


# --- F1's real oracle: an edit under a footprint, patched --------------


def test_an_edit_under_a_footprint_patches_to_a_fresh_full_render(sprite_install):
    """**F1 is a derived claim -- that Flat's per-tile dirty rects need no
    widening for icons -- and this is what says so or not.** The edited tile
    sits UNDER the 3x3 building's footprint, or the check fires vacuously
    against terrain no icon ever covered.

    patch() routes the per-tile rect through _composite_rect on a fresh
    scratch, terrain first, so an icon lands wherever a full render would put
    it. If that were wrong, the patched canvas and a fresh one would disagree
    exactly over the building.
    """
    scn = _scenario(_units())
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    cache = render_cache.FlatChunkCache(scn, tile_px, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk

    tx, ty = 2, 2  # inside the 3x3 building at (2, 2), which spans tiles 1..3
    building = next(u for units in scn.unit_manager.units for u in units if u.unit_const == BIG_CONST)
    bx0, bx1, by0, by1 = render.unit_tile_bounds(building, MAP_W, MAP_H)
    assert bx0 <= tx < bx1 and by0 <= ty < by1, "the edited tile is not under a footprint"
    scn.map_manager.get_tile(tx, ty).terrain_id = 6
    cache.patch_rects([(tx * tile_px, ty * tile_px, (tx + 1) * tile_px, (ty + 1) * tile_px)])

    fresh = render_cache.FlatChunkCache(scn, tile_px, sprites=True)
    assert np.array_equal(
        cache.render_rect(0, 0, canvas_w, canvas_h),
        fresh.render_rect(0, 0, canvas_w, canvas_h),
    )


# --- icons and draws must go stale together ---------------------------


@pytest.mark.parametrize("mip", [0, -1])
def test_a_filter_change_drops_icons_in_lockstep_with_draws(sprite_install, mip):
    """Icons are keyed by ROW INDEX into _flat_unit_draws()' bboxes, so a
    filter change that rebuilt the draws without the icons would shift every
    later row and paint each surviving unit with its neighbour's sprite --
    which no existing test catches, since the pixels stay plausible.

    Level 0 is checked as well as a non-zero one on purpose: it is where
    _level_unit_draws()' own shape diverges (`mip == 0` returns the public
    self.unit_draws), so a copy-the-neighbour implementation of _level_icons()
    would leave exactly that level stale.
    """
    scn = _scenario(_units())
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    cache = render_cache.FlatChunkCache(scn, tile_px, sprites=True)
    before = _stitched(cache, mip)

    # Hiding GAIA drops row 0 specifically, which shifts EVERY later row --
    # the exact shape a stale icon layer would mis-key on.
    hide_gaia = UnitFilter(show_gaia=False)
    cache.set_unit_filter(hide_gaia)
    filtered = render_cache.FlatChunkCache(scn, tile_px, sprites=True, unit_filter=hide_gaia)

    assert not np.array_equal(before, _stitched(cache, mip)), "the filter changed nothing"
    assert np.array_equal(_stitched(cache, mip), _stitched(filtered, mip))


def test_turning_sprites_on_live_matches_a_cache_built_with_them(sprite_install):
    """set_sprites_enabled() on a warmed cache must reach the already-composited
    chunk PIXELS, not just the flag -- units are baked into them, so without the
    eviction the toggle appears to do nothing until the user scrolls somewhere
    uncached."""
    scn = _scenario(_units())
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    cache = render_cache.FlatChunkCache(scn, tile_px, sprites=False)
    cache.render_rect(0, 0, *cache.canvas_dims())
    cache.set_sprites_enabled(True)

    built_on = render_cache.FlatChunkCache(scn, tile_px, sprites=True)
    assert np.array_equal(_stitched(cache), _stitched(built_on))
    assert not np.array_equal(_stitched(cache), _full(scn, with_sprites=False))


# --- Flat's own facing zero point -------------------------------------


def _frame_colour(frame_index: int) -> tuple[int, int, int]:
    """The solid RGB build_sld() gives that frame. Read through
    _native_frame(), i.e. the real decode, so a frame the file does not
    actually hold shows up as a crash rather than as a plausible expectation."""
    main = unit_sprites._native_frame(ANGLE_FILE, frame_index)[0]
    return tuple(int(v) for v in main[0, 0, :3])


@pytest.mark.parametrize("rotation", [0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
def test_a_flat_icon_draws_the_frame_the_flat_zero_point_selects(sprite_install, rotation):
    """**The wiring check for FLAT_ANGLE_ZERO_OFFSET_DEG, and the only test
    that goes red if icon_for() stops passing it.** Everything in
    tests/test_unit_sprites.py exercises angle_index()/_frame_for() directly
    with the constant handed in, so all of it stays green while Flat still
    renders at the isometric zero point, which is exactly the bug this
    change fixes.

    The expectation is DERIVED from the constant rather than hardcoded or
    monkeypatched, so re-measuring the offset against the game moves this test
    with it and no clear_caches() dance is needed. team_index 0 is GAIA, whose
    tint is an identity multiply, so the icon's pixels are the decoded frame's
    own colour with no tint arithmetic in between.

    All four cardinals, because the shift is a rotation of the whole angle set:
    a mutation that happened to agree at one of them is not a fix.
    """
    iso_frame = unit_sprites.angle_index(rotation, ANGLE_COUNT)
    flat_frame = unit_sprites.angle_index(
        rotation, ANGLE_COUNT, unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG
    )
    assert iso_frame != flat_frame, (
        f"the two zero points pick the same frame at rotation {rotation}, so this "
        f"cannot tell them apart"
    )
    assert _frame_colour(flat_frame) != _frame_colour(iso_frame), (
        "the fixture's two frames are the same colour, so the pixels below prove nothing"
    )

    icon = unit_sprites.icon_for(ANGLE_CONST, rotation, 0, 32, 32)
    assert icon is not None, "the icon resolved to nothing, so this proves nothing"
    got = tuple(int(v) for v in icon.rgba[0, 0, :3])
    assert got == _frame_colour(flat_frame), (
        f"rotation {rotation} drew frame colour {got}; the flat zero point selects frame "
        f"{flat_frame} ({_frame_colour(flat_frame)}) and the isometric one frame "
        f"{iso_frame} ({_frame_colour(iso_frame)})"
    )
