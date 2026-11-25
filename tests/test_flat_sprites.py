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
the unit-free blank template, on which anything here would run vacuous. Every
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
- Drop the `flat_art()` call in `icon_for()` (GH #151, hand `unit_const` and
  `rotation` straight down): RED, 5 of 6, via the wall and gate checks at the
  bottom of this file. Shape 2, the tower, stays green by design. Recorded
  2026-10-19; every flat_art test in tests/test_unit_sprites.py stays green
  under it, for the same reason as the arm above.
- Turn the wrong way (`FLAT_RUN_STEPS = 3`): RED via the gate check here and
  the first-principles turn checks in tests/test_unit_sprites.py; the wall
  check here stays green, since it derives its frame from flat_wall_shape().
- Drop the de-project branch in `icon_for()` (every cliff contain-fits, TASK-350):
  RED, 6 of 8, via the thin-straight checks, the Enhanced check and the scope
  check's own cliff draw. The frame-4 controls stay green by design. Recorded
  2026-11-25.
- Drop the `"deproject"` cache-key tag: RED via the scope check only (the
  non-cliff const is served the cliff's cached icon). Drop the cliff-membership
  test from the branch: RED, the same check. Size x2 art with x1's tile
  (`scale = tile / (NATIVE_TILE_W / 2)`): RED via the Enhanced check, the run
  twice as thick (ink 2046 px against 1020) and off centre. De-project every
  cliff frame (drop the frame condition): RED, 2 of 2, via the frame-4
  controls. Shift the hotspot `_deprojected_icon()` passes by a quarter tile:
  RED, 4 of 4, via the thin-straight checks' placement assertion, added for
  exactly this (the axis and reach assertions alone stayed green). Recorded
  2026-11-25.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import pytest

from descape import (
    asset_source,
    flat_deproject,
    iso_geometry,
    render_cache,
    render_flat,
    render_paint,
    render_units,
    unit_sprites,
)
from descape.terrain_palette import BUILDING_TILE_SPANS, PLAYER_COLORS
from descape.unit_filter import UnitFilter

import conftest
from test_unit_sprites import _FRAME_HEADER, _HEADER, CONST, FILE_NAME, MAGIC, MAIN, bc1_solid, build_sld

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
# Non-composite on purpose: _build_icon() then takes its single-piece path,
# so the icon's pixels are that one frame's, uncomposited.
ANGLE_CONST = CONST + 2
ANGLE_FILE = "t_facing_x1"
ANGLE_COUNT = 16

# GH #151's wiring checks need REAL consts, since flat_art() keys on the wall
# set and gate_orientation's committed tables: Stone Wall, and the stone gate
# 64 (4x1) with 659, its next orientation sibling. Each gets its own file.
WALL_CONST, WALL_FILE = 117, "t_wall_x1"
GATE_CONST, GATE_FILE = 64, "t_gate_a_x1"
GATE_TURNED, GATE_TURNED_FILE = 659, "t_gate_b_x1"


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
    (graphics / f"{WALL_FILE}.sld").write_bytes(build_sld(5, canvas=unit_sprites.NATIVE_TILE_W))
    (graphics / f"{GATE_FILE}.sld").write_bytes(build_sld(1, colour_base=10))
    (graphics / f"{GATE_TURNED_FILE}.sld").write_bytes(build_sld(1, colour_base=20))

    def entry(const):
        piece = {"unit_id": const, "file_name": FILE_NAME, "angle_count": 4,
                 "frame_count": 1, "dx": 0, "dy": 0, "parent": True}
        below = {**piece, "unit_id": const + 100, "dy": unit_sprites.NATIVE_TILE_W}
        del below["parent"]
        return {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                "mirroring_mode": 6, "frame_count": 1, "pieces": [piece, below]}

    facing = {"graphic_id": 2, "file_name": ANGLE_FILE, "angle_count": ANGLE_COUNT,
              "mirroring_mode": 6, "frame_count": 1}

    def single(file_name, angle_count, **extra):
        return {"graphic_id": 3, "file_name": file_name, "angle_count": angle_count,
                "mirroring_mode": 0, "frame_count": 1, **extra}

    table = {
        CONST: entry(CONST), BIG_CONST: entry(BIG_CONST), ANGLE_CONST: facing,
        WALL_CONST: single(WALL_FILE, 5, rotation_is_variant=True),
        GATE_CONST: single(GATE_FILE, 1, rotation_is_variant=True),
        GATE_TURNED: single(GATE_TURNED_FILE, 1, rotation_is_variant=True),
    }
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    monkeypatch.setitem(BUILDING_TILE_SPANS, BIG_CONST, BIG_SPAN)
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
        render_paint.tile_pixels_for_map(MAP_W, MAP_H)
    ).values())


def _full(scn, *, with_sprites):
    """A full Flat render the chunk path is compared against -- deliberately
    render_terrain + overlay_units, the INDEPENDENT implementation, never
    composite_rect_flat. See composite_rect_flat()'s own docstring on why
    sharing one would make the comparison a tautology."""
    return render_flat.overlay_units(render_flat.render_terrain(scn), scn, with_sprites=with_sprites)


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
            bounds = render_units.unit_tile_bounds(unit, mm.map_width, mm.map_height)
            if bounds is None:
                continue
            tx0, tx1, ty0, ty1 = bounds
            # Shifted by the same off-grid offset _draw_unit() applies
            # (Stage 1, 2026-09-17). This fixture places units at INTEGER
            # coordinates on purpose, and an integer coordinate is a grid
            # vertex, not a tile centre -- so those marks legitimately sit
            # half a tile up-left of their own tile and this rect has to
            # follow them or the no-widening assertion tests the wrong box.
            dx, dy = render_units.unit_paint_offset(unit)
            ox, oy = round(dx * tile_px), round(dy * tile_px)
            rects.append((
                tx0 * tile_px + ox, ty0 * tile_px + oy,
                tx1 * tile_px + ox, ty1 * tile_px + oy,
            ))
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
           iso_geometry.mip_tile_px_candidates(render_paint.tile_pixels_for_map(MAP_W, MAP_H)).items()}[tile_px]

    terrain_only = _stitched(render_cache.FlatChunkCache(scn, render_paint.tile_pixels_for_map(MAP_W, MAP_H),
                                                         with_units=False), mip)
    with_icons = _stitched(render_cache.FlatChunkCache(scn, render_paint.tile_pixels_for_map(MAP_W, MAP_H),
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
    tile_px = render_paint.tile_pixels_for_map(MAP_W, MAP_H)
    # Bare terrain, NOT the coloured-mark render: a mark fills the whole cell,
    # so diffing against it would report the whole cell as changed and say
    # nothing about where the icon's own ink landed.
    terrain_only = render_flat.render_terrain(scn)
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
    tile_px = render_paint.tile_pixels_for_map(MAP_W, MAP_H)
    cache = render_cache.FlatChunkCache(scn, tile_px, sprites=True)
    assert cache._level_icons(0), "no unit resolved to an icon, so this proves nothing"

    seams = {x for x, _y, x1, _y1 in _footprint_rects(scn, tile_px) for x in (x, x1)}
    assert any(s % CHUNK > 0 for s in seams if s % CHUNK), "no footprint crosses a chunk seam"

    assert np.array_equal(_stitched(cache), _full(scn, with_sprites=True))


def test_sprites_off_is_byte_identical_to_the_pre_sprite_render(sprite_install):
    """At every mip level. With icons=None composite_rect_flat's painted code
    path is unchanged bytes, so this is structural rather than a coincidence
    -- but the wiring above it (a cache built with sprites=False still asking
    for a layer, say) is not, and that is what this catches."""
    scn = _scenario(_units())
    tile_px = render_paint.tile_pixels_for_map(MAP_W, MAP_H)
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
    tile_px = render_paint.tile_pixels_for_map(MAP_W, MAP_H)
    cache = render_cache.FlatChunkCache(scn, tile_px, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk

    tx, ty = 2, 2  # inside the 3x3 building at (2, 2), which spans tiles 1..3
    building = next(u for units in scn.unit_manager.units for u in units if u.unit_const == BIG_CONST)
    bx0, bx1, by0, by1 = render_units.unit_tile_bounds(building, MAP_W, MAP_H)
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
    tile_px = render_paint.tile_pixels_for_map(MAP_W, MAP_H)
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
    tile_px = render_paint.tile_pixels_for_map(MAP_W, MAP_H)
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


# --- walls and gates turned onto the square grid (GH #151) -------------


def _file_colour(file_name: str, frame_index: int) -> tuple[int, int, int]:
    main = unit_sprites._native_frame(file_name, frame_index)[0]
    return tuple(int(v) for v in main[0, 0, :3])


def _centre_colour(draw: unit_sprites.SpriteDraw) -> tuple[int, int, int]:
    h, w = draw.rgba.shape[:2]
    return tuple(int(v) for v in draw.rgba[h // 2, w // 2, :3])


@pytest.mark.parametrize("shape", range(5))
def test_a_flat_wall_icon_draws_the_turned_shape_and_iso_keeps_the_stored_one(sprite_install, shape):
    """**The wiring check for flat_art()'s wall half.** The frame colours say
    which shape was drawn, so this goes red if icon_for() stops calling it,
    while sprite_for() is the iso control that must not move."""
    turned = unit_sprites.flat_wall_shape(shape)
    colours = {_file_colour(WALL_FILE, k) for k in range(5)}
    assert len(colours) == 5, "the wall's five frames are not distinct colours"

    icon = unit_sprites.icon_for(WALL_CONST, float(shape), 0, 32, 32)
    assert icon is not None, "the icon resolved to nothing, so this proves nothing"
    assert _centre_colour(icon) == _file_colour(WALL_FILE, turned), (
        f"stored shape {shape}: Flat should draw frame {turned}"
    )
    sprite = unit_sprites.sprite_for(WALL_CONST, float(shape), 0, 32)
    assert sprite is not None
    assert _centre_colour(sprite) == _file_colour(WALL_FILE, shape), "the iso view must keep frame k"


def test_a_flat_gate_icon_draws_its_turned_sibling_in_its_own_footprint(sprite_install):
    """**The wiring check for flat_art()'s gate half**: 64's 4x1 footprint
    holds 659's art in Flat, while iso still draws 64's own."""
    from descape import gate_orientation

    assert gate_orientation.cycle_const(GATE_CONST, 1) == GATE_TURNED
    assert _file_colour(GATE_FILE, 0) != _file_colour(GATE_TURNED_FILE, 0)

    icon = unit_sprites.icon_for(GATE_CONST, 0.0, 0, 4 * 32, 32)
    assert icon is not None, "the icon resolved to nothing, so this proves nothing"
    assert icon.rgba.shape[1] <= 4 * 32 and icon.rgba.shape[0] <= 32, "the icon left the placed footprint"
    assert _centre_colour(icon) == _file_colour(GATE_TURNED_FILE, 0)
    sprite = unit_sprites.sprite_for(GATE_CONST, 0.0, 0, 32)
    assert sprite is not None
    assert _centre_colour(sprite) == _file_colour(GATE_FILE, 0)


# --- thin cliff straights de-projected onto the grid (TASK-350) --------

# Two synthetic cliff consts sharing one 24-frame file: a 3x1 and a 1x3, as a
# family's suffix 3 and 2 pieces are, plus a non-cliff const on the same file.
CLIFF_X, CLIFF_Y, NOT_CLIFF = CONST + 10, CONST + 11, CONST + 12
CLIFF_FILE = "t_cliff_x1"
# Canvas in 4 px blocks, hotspot at its centre. A stroke 2 blocks thick along
# the iso +x screen direction (2 right per 1 up) or +y (2 right per 1 down),
# about 3.1 tiles long, so a de-projected run spans its whole 3-tile footprint.
_CLIFF_BLOCKS_W, _CLIFF_BLOCKS_H = 40, 24
_STROKE_FRAMES = {10: -1, 4: -1, 8: 1}  # frame -> rows per step along +x


def _stroke_layer(mask: np.ndarray, rgb565: int) -> bytes:
    """A MAIN layer drawing a solid block wherever `mask` holds, skipping the rest."""
    flat, cmds, i = mask.ravel(), [], 0
    while i < flat.size:
        skip = draw = 0
        while i < flat.size and not flat[i] and skip < 255:
            skip, i = skip + 1, i + 1
        while i < flat.size and flat[i] and draw < 255:
            draw, i = draw + 1, i + 1
        cmds.append(bytes((skip, draw)))
    h, w = mask.shape
    body = struct.pack("<4H2B", 0, 0, 4 * w, 4 * h, 0, 1) + struct.pack("<H", len(cmds)) + b"".join(cmds)
    body += bc1_solid(rgb565) * int(flat.sum())
    return struct.pack("<I", 4 + len(body)) + body


def _cliff_sld(k: int = 1) -> bytes:
    """The cliff file, or at k=2 its _x2 copy: the same art at exactly twice the size."""
    w, h = 4 * k * _CLIFF_BLOCKS_W, 4 * k * _CLIFF_BLOCKS_H
    out = bytearray(_HEADER.pack(MAGIC, 4, 24, 0, 16, 0))
    for frame in range(24):
        mask = np.zeros((_CLIFF_BLOCKS_H, _CLIFF_BLOCKS_W), dtype=bool)
        sense = _STROKE_FRAMES.get(frame)
        if sense is None:
            mask[8:16, 12:28] = True
        else:
            for t in range(-9, 10):
                row, col = _CLIFF_BLOCKS_H // 2 + sense * t, _CLIFF_BLOCKS_W // 2 + 2 * t
                mask[row, max(0, col - 1) : col + 1] = True
        mask = mask.repeat(k, axis=0).repeat(k, axis=1)
        out += _FRAME_HEADER.pack(w, h, w // 2, h // 2, MAIN, 0, frame)
        out += _stroke_layer(mask, 0x0800 * (frame + 1))
        out += bytes([0xAA]) * ((4 - len(out)) % 4)
    return bytes(out)


@pytest.fixture
def cliff_install(sprite_install, tmp_path, monkeypatch):
    from descape import terrain_palette

    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    (graphics / f"{CLIFF_FILE}.sld").write_bytes(_cliff_sld())
    (graphics / f"{CLIFF_FILE[:-3]}_x2.sld").write_bytes(_cliff_sld(2))
    cliff = {"graphic_id": 4, "file_name": CLIFF_FILE, "angle_count": 24, "mirroring_mode": 0,
             "frame_count": 1, "rotation_is_variant": True}
    table = {**unit_sprites.graphic_map(), CLIFF_X: cliff, CLIFF_Y: cliff, NOT_CLIFF: cliff}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    cliffs = unit_sprites._CLIFF_VARIANT_BY_GAME["aoe2"]
    monkeypatch.setitem(unit_sprites._CLIFF_VARIANT_BY_GAME, "aoe2", cliffs | {CLIFF_X, CLIFF_Y})
    for const, span in ((CLIFF_X, (3, 1)), (CLIFF_Y, (1, 3)), (NOT_CLIFF, (3, 1))):
        monkeypatch.setitem(terrain_palette.OBJECT_TILE_SPANS, const, span)
    unit_sprites.clear_caches()
    return conftest.load_verify_module("verify_flat_facing")


def _contain_fit(const: int, frame: int, fw: int, fh: int):
    entry = unit_sprites.graphic_map()[const]
    return unit_sprites._build_icon(
        const, entry, float(frame), unit_sprites.TEAM_COLORS[0], fw, fh, unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG,
    )


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize(("const", "frame", "axis"), [(CLIFF_X, 10, 0.0), (CLIFF_Y, 8, 90.0)])
def test_a_thin_cliff_straight_lies_along_its_flat_footprint(cliff_install, const, frame, axis, smooth):
    """**The wiring check for the de-projection.** Contain-fit leaves the iso
    stroke a diagonal dash (the control below); the icon_for() one runs along
    its footprint's long side and spans most of it, at both resample paths."""
    unit_sprites.set_smooth_shrink(smooth)
    span = (3, 1) if const == CLIFF_X else (1, 3)
    fw, fh = span[0] * 32, span[1] * 32
    fitted = _contain_fit(const, frame, fw, fh)
    assert fitted is not None
    assert cliff_install.axis_gap_deg(cliff_install.principal_axis_deg(fitted.rgba[..., 3]), axis) > 20, (
        "the contain-fitted stroke already lies along the footprint, so this cannot tell the two apart"
    )

    icon = unit_sprites.icon_for(const, float(frame), 0, fw, fh)
    assert icon is not None, "the icon resolved to nothing, so this proves nothing"
    assert icon.rgba.shape[:2] == (fh, fw), "a de-projected icon is its whole footprint"
    alpha = icon.rgba[..., 3]
    got = cliff_install.principal_axis_deg(alpha)
    assert cliff_install.axis_gap_deg(got, axis) <= 5.0, f"frame {frame}: axis {got:.1f}, want {axis}"
    long_axis = 0 if const == CLIFF_X else 1
    reach = np.flatnonzero((alpha > 0).any(axis=long_axis))  # the columns of an x-run, rows of a y-run
    assert reach.size >= 0.8 * (fw, fh)[long_axis], f"the run covers {reach.size} of {(fw, fh)[long_axis]} px"
    # Placement: the shipped icon sits where the native frame's own de-projection does, scaled to 32 px tiles.
    want_x, want_y = (c * 32 / (unit_sprites.NATIVE_TILE_W / 2) for c in _centroid(_native_deprojection(const, frame, span)))
    got_x, got_y = _centroid(alpha)
    assert abs(got_x - want_x) <= 1.5 and abs(got_y - want_y) <= 1.5, (
        f"ink centred at ({got_x:.1f}, {got_y:.1f}), the native de-projection at ({want_x:.1f}, {want_y:.1f})"
    )


def _native_deprojection(const: int, frame: int, span: tuple[int, int]) -> np.ndarray:
    """The frame de-projected at native scale (no resize) onto its footprint: the alpha the icon scales."""
    entry = unit_sprites.graphic_map()[const]
    (piece,) = unit_sprites._native_pieces_for(const, entry, float(frame), unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG)
    tile = unit_sprites.NATIVE_TILE_W // 2
    out = flat_deproject.deprojected(piece.main, piece.hotspot_x, piece.hotspot_y, tile, tile, span[0] * tile, span[1] * tile)
    return out[..., 3]


def _centroid(alpha: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(alpha)
    return float(xs.mean()) + 0.5, float(ys.mean()) + 0.5


def test_a_de_projected_cliff_at_enhanced_draws_the_x2_art_in_the_same_place(cliff_install):
    """At Enhanced a footprint whose tiles enlarge x1 (64 px against x1's
    48 px half diamond) draws from x2 (GH #241). The x2 copy is the same art
    at twice the size, so the icon must cover what x1's does: an x2 frame
    resampled with x1's tile size would draw a run twice as long, off centre."""
    entry = unit_sprites.graphic_map()[CLIFF_X]
    assert unit_sprites._native_pieces_for(CLIFF_X, entry, 10.0, x2=True), "no x2 art, so this proves nothing"
    x1 = unit_sprites.icon_for(CLIFF_X, 10.0, 0, 3 * 64, 64)
    unit_sprites.set_hi_res(True)
    x2 = unit_sprites.icon_for(CLIFF_X, 10.0, 0, 3 * 64, 64)
    assert x1 is not None and x2 is not None
    assert not np.array_equal(x1.rgba, x2.rgba), "the two draws are byte-identical, so x2 was not drawn"
    # x1's nearest 4/3 enlargement stipples where x2's shrink is solid, so compare the run, not pixels.
    runs = []
    for draw in (x1, x2):
        ink = draw.rgba[..., 3] > 0
        rows = np.nonzero(ink)[0]
        runs.append((rows.mean(), ink.sum() / ink.any(axis=0).sum(), ink.any(axis=0).sum()))
    (row1, thick1, len1), (row2, thick2, len2) = runs
    assert abs(row1 - row2) <= 2.0, f"run centre row {row1:.1f} at x1, {row2:.1f} at x2"
    assert 2 / 3 <= thick2 / thick1 <= 3 / 2, f"run thickness {thick1:.1f} px at x1, {thick2:.1f} at x2"
    assert min(len1, len2) >= 0.9 * 3 * 64, f"the run covers {len1} px at x1, {len2} at x2, of {3 * 64}"


@pytest.mark.parametrize("const", [CLIFF_X, CLIFF_Y])
def test_a_cliff_frame_outside_the_thin_straights_keeps_its_contain_fit(cliff_install, const):
    """Frame 4 is a tall straight's: it has the same diagonal stroke as frame
    10, and must still draw exactly as _build_icon() does."""
    assert 4 not in unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES
    fw, fh = (96, 32) if const == CLIFF_X else (32, 96)
    icon = unit_sprites.icon_for(const, 4.0, 0, fw, fh)
    want = _contain_fit(const, 4, fw, fh)
    assert icon is not None and want is not None
    assert np.array_equal(icon.rgba, want.rgba)
    entry = unit_sprites.graphic_map()[const]
    deprojected = unit_sprites._deprojected_icon(const, entry, 4.0, unit_sprites.TEAM_COLORS[0], fw, fh)
    assert deprojected is not None and deprojected.rgba.shape != want.rgba.shape, "the two draws coincide"


def test_a_non_cliff_const_on_frame_10_is_not_de_projected(cliff_install):
    """Scope: the branch keys on the active game's cliff set, not on the art
    or the frame. NOT_CLIFF shares CLIFF_X's file, span and frame, and is
    drawn right after it, so a cache key without the de-project tag would
    hand it the cliff's icon."""
    assert unit_sprites.icon_for(CLIFF_X, 10.0, 0, 96, 32).rgba.shape[:2] == (32, 96)
    icon = unit_sprites.icon_for(NOT_CLIFF, 10.0, 0, 96, 32)
    want = _contain_fit(NOT_CLIFF, 10, 96, 32)
    assert icon is not None and want is not None
    assert np.array_equal(icon.rgba, want.rgba)
    # An ANGLE unit whose facing resolves to index 10 is a facing, not a cliff frame.
    rotation = 10 * 2 * math.pi / ANGLE_COUNT
    angle_entry = unit_sprites.graphic_map()[ANGLE_CONST]
    assert unit_sprites._frame_key(ANGLE_CONST, angle_entry, rotation, unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG) == (10,)
    angle_icon = unit_sprites.icon_for(ANGLE_CONST, rotation, 0, 96, 32)
    angle_want = unit_sprites._build_icon(
        ANGLE_CONST, angle_entry, rotation, unit_sprites.TEAM_COLORS[0], 96, 32, unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG,
    )
    assert np.array_equal(angle_icon.rgba, angle_want.rgba)
