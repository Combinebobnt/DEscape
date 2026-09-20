"""An edit near a sprite-bearing tile must repaint the whole sprite (P3-g5).

`merge_sprite_bboxes` (P3-g3) fixed the BYSTANDER half: a sprite no longer clips
at a chunk boundary. This module covers the other half. `_dirty_screen_bbox`
answers "what pixels did this edit invalidate", and it never consulted
`building_bboxes` at all -- it dilated the dirty tiles by
`UNIT_FOOTPRINT_MAX_RADIUS` and unioned per-tile swept bounds, which model the
terrain diamond, the skirt and the contact shadow and NO unit pixel extent. So
an edit beside a big building under-repainted it and left a stale fragment.

The gap is far wider vertically than horizontally, and it grows as
`elev_step_pct` shrinks: the swept bbox used to carry `(max_elev - e) * elev_step` of
free slack, which is what accidentally covered the horizontal case. Tall
sprites out-reach it upward by up to 230px at `tile_px=64`.

**Nothing here can verify that the committed `unit_sprites.MAX_SPRITE_REACH_*`
constants actually bound real game assets.** This module's sprite is synthetic
bytes on a tmp install, and it monkeypatches those constants to match, so the
fixture and the widening stay consistent by construction (see the fixture's own
note on why that is required rather than tidy). What keeps the constants honest
against a game patch is `tools/scan_sprite_reach.py` and the `corpus` test at
the bottom of this file, which needs a real install and skips without one.

Tests are ordered cheapest-first. Test 2 in particular must stay ABOVE the
oracle: it is the one that says "the fixture is not vacuous", and reading that
failure first is what stops a future session debugging working code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from descape import asset_source, iso_geometry, render, unit_sprites
from descape.elevation_tools import set_tile_elevation
from descape.render import dirty_screen_bbox_iso, render_terrain_iso_with_proj
from descape.render_cache import IsoChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

import conftest
from test_unit_sprites import CONST, FILE_NAME, build_sld

# 12 tiles wide at native scale, so the sprite reaches 6 whole tiles past its
# hotspot on every side -- see the fixture for why this specific size.
SPRITE_CANVAS = 12 * unit_sprites.NATIVE_TILE_W

REACH_NAMES = (
    "MAX_SPRITE_REACH_LEFT",
    "MAX_SPRITE_REACH_UP",
    "MAX_SPRITE_REACH_RIGHT",
    "MAX_SPRITE_REACH_DOWN",
)

# Elevation 14 everywhere, edited to 15. NOT arbitrary: the swept bbox used to carry
# (max_elev - e) * elev_step of free vertical slack, so a sprite low on the map
# is covered for the wrong reason and the whole module passes vacuously. At
# max_elev that slack is zero and the widening is the only thing holding the
# bbox open. +-1 also keeps _elevation_tile_recursion's propagation to a small
# blob, which matters because a blob spanning k tiles in d = y - x eats k *
# half_h of the margin this measures.
BASE_ELEVATION = 14
EDIT_ELEVATION = 15
EDIT_X = EDIT_Y = 60
# The only heights the edit ever paints a sprite at. Coverage is asserted here,
# not at min_elev/max_elev: the bbox sweeps the observed pre/post range now
# (bbox-elev-sweep plan), and the scenario never holds any other height.
OBSERVED_ELEVATIONS = (BASE_ELEVATION, EDIT_ELEVATION)

# The coarser mip the bottom test patches alongside level 0. This fixture
# enumerates [-2, -1, 0, 1] at tile_px 64, so -1 is tile_px 32: an exact
# halving, which is what makes is_exact_mip's shrink-by-the-same-ratio
# argument the thing under measurement rather than a rounding question.
COARSER_LEVEL = -1


@dataclass
class Unit:
    """The four attributes `_units_by_tile`/`sprite_draws_by_anchor`/
    `unit_tile_bounds` actually read. Duck-typed rather than a real
    genieutils unit, matching tests/test_sprite_chunks.py's own posture."""

    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """A tmp install holding one sprite much bigger than its tile, with the
    reach constants pinned to match it.

    Copied from tests/test_sprite_chunks.py rather than shared: a fixture in a
    test module does not cross modules, and this one's needs differ from that
    one's in two ways that matter.

    **The canvas is 12 * NATIVE_TILE_W, not 4.** build_sld puts the hotspot at
    the canvas centre, so this is a native reach of 576 on all four sides. That
    has to out-reach the elevation sweep to prove anything at all.

    **And the four MAX_SPRITE_REACH_* constants are pinned to that same 576.**
    Not tidiness -- required. The real constants are 404/650/424/200, so an
    unpinned fixture would reach 172px further left, 152px further right and
    376px further down than a CORRECT widening is ever asked to cover. The
    oracle would go red against working code, and the next session would spend
    a debug cycle on it. Pinned, fixture and widening agree by construction and
    all four directions are exercised at once.
    """
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
    # No SPRITES_ENABLED monkeypatch here as of P3-g's toggle: sprites are now
    # per-call (dirty_screen_bbox_iso's with_sprites=) and per-cache
    # (IsoChunkCache(sprites=)), so patching the module constant would reach
    # nothing. Leaving an inert one in place would be worse than useless --
    # every call below would silently compute the sprites-OFF bbox and this
    # module's non-vacuity halves would pass while measuring nothing, which is
    # exactly what test_zeroing_the_reaches... exists to rule out.
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _fixture_scenario(ux: float = float(EDIT_X), uy: float = float(EDIT_Y)):
    """The 120x120 blank template, flat at BASE_ELEVATION, with one
    sprite-bearing unit on it.

    Player 1, so TEAM_COLORS[1] tints the sprite blue and it cannot be confused
    with terrain. The real template rather than a 12x12 synthetic map, and NOT
    shrinkable to save time: at tile_px=64 a 12x12 map puts the sprite's top at
    y=-32, where the canvas clamp swallows the whole vertical assertion. At
    (60, 60) on 120x120 nothing clamps on any side.
    """
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = BASE_ELEVATION
    scenario.unit_manager.units[1].append(Unit(ux, uy, CONST))
    return scenario


def _elevation_grid(mm) -> np.ndarray:
    return np.array(
        [[int(mm.get_tile(x, y).elevation) for x in range(mm.map_width)] for y in range(mm.map_height)]
    )


def _edit_and_dirty(mm, ex: int, ey: int, elevation: int):
    """The real single-tile elevation edit, returning the terrain INDICES it
    actually changed -- copied from tests/test_sloped_edit.py, where the same
    reasoning applies: one click propagates through _elevation_tile_recursion
    to many tiles, and the propagated set is what a stroke hands _apply_dirty."""
    before = _elevation_grid(mm)
    set_tile_elevation(mm, ex, ey, elevation)
    after = _elevation_grid(mm)
    changed = {(int(x), int(y)) for y, x in zip(*np.nonzero(before != after), strict=True)}
    assert changed, "set_tile_elevation produced no change -- fixture is broken"
    return [i for i, tile in enumerate(mm.terrain) if (tile.x, tile.y) in changed], changed


def _edited_scenario(ux: float = float(EDIT_X), uy: float = float(EDIT_Y)):
    """(scenario, dirty_indices, elevations, proj), post-edit.

    elevations/proj are PRE-edit -- the state the viewer's cache is in when an
    edit lands. dirty_screen_bbox_iso is what brings elevations forward, in
    place, which is exactly the contract under test here.

    Via elevations_and_proj() rather than a full render: it is the same call
    ViewerWindow makes to size its cache, so this is the production path, and
    it keeps every test but the oracle off the ~0.5s composite.
    """
    scenario = _fixture_scenario(ux, uy)
    elevations, proj = render.elevations_and_proj(scenario)
    dirty, _changed = _edit_and_dirty(scenario.map_manager, EDIT_X, EDIT_Y, EDIT_ELEVATION)
    return scenario, dirty, elevations, proj


def _sprite_rect(scenario, proj, elevation: int) -> tuple[int, int, int, int]:
    """The real canvas rect sprite_draws_by_anchor would paint, with the whole
    map pinned at `elevation`. Ground truth, pulled from the production
    function rather than re-derived -- re-deriving it here would let the same
    mistake pass on both sides."""
    mm = scenario.map_manager
    elevations = np.full((mm.map_height, mm.map_width), elevation, dtype=np.int64)
    layer = render.sprite_draws_by_anchor(scenario, proj, elevations)
    assert len(layer.by_anchor) == 1, f"expected exactly one sprite, got {layer.by_anchor.keys()}"
    draw, ax, ay = next(iter(layer.by_anchor.values()))[0]
    h, w = draw.rgba.shape[:2]
    x0, y0 = ax - draw.hotspot_x, ay - draw.hotspot_y
    return x0, y0, x0 + w, y0 + h


def _clamped(rect, proj) -> tuple[int, int, int, int]:
    """A rect clipped to the canvas the bbox itself is clamped to -- the same
    correction tests/test_sloped_edit.py's ring check makes, for the same
    reason: a sprite hanging off the canvas can only be required to be covered
    where the canvas actually exists."""
    canvas_w, canvas_h = render._canvas_pixel_dims(proj)
    x0, y0, x1, y1 = rect
    return max(0, x0), max(0, y0), min(canvas_w, x1), min(canvas_h, y1)


def _covers(bbox, rect) -> bool:
    return bbox[0] <= rect[0] and bbox[1] <= rect[1] and bbox[2] >= rect[2] and bbox[3] >= rect[3]


def _full_render_at_level(cache, monkeypatch, scenario) -> np.ndarray:
    """A fresh whole-canvas render at COARSER_LEVEL's tile_px, the ground
    truth the cache's coarse level is compared against. Same technique
    tests/test_mip_geometry.py's patch-at-a-non-reference-level check uses.

    monkeypatch.context() rather than a bare setattr + undo(): undo() is
    all-or-nothing and would also revert sprite_install's graphic_map and
    MAX_SPRITE_REACH_* patches, silently turning every later assertion in the
    test into a measurement of the real constants against a synthetic sprite.
    """
    with monkeypatch.context() as m:
        m.setattr(render, "tile_pixels_for_map", lambda w, h: cache.mip_tile_px(COARSER_LEVEL))
        full, _elev, _proj = render_terrain_iso_with_proj(scenario, with_sprites=True)
    return full


def _zero_the_reaches(monkeypatch) -> None:
    """The in-suite mutation knob: reaches at 0 leaves the widening block
    running but contributing nothing beyond its 1px rounding slack."""
    for name in REACH_NAMES:
        monkeypatch.setattr(unit_sprites, name, 0)


# --- cheap checks, deliberately above the oracle ------------------------


def test_the_widening_grows_the_bbox_and_covers_a_sprite_the_plain_one_misses(
    sprite_install, monkeypatch
):
    """Test 2. Non-vacuity, stated against ground truth rather than re-derived.

    Two halves. First, the bbox must actually GROW on all four sides versus the
    same call with the reaches zeroed -- if it does not, every other assertion
    in this module is being satisfied by the pre-existing dilation and proves
    nothing. Second, the defect itself: the real sprite rect from
    sprite_draws_by_anchor must fall OUTSIDE the un-widened bbox and INSIDE the
    widened one.

    Both bbox calls get elevations.copy(): dirty_screen_bbox_iso mutates its
    array in place (deliberately -- Risk #6), so a shared one would leave the
    second call measuring an already-advanced snapshot.
    """
    scenario, dirty, elevations, proj = _edited_scenario()

    widened = dirty_screen_bbox_iso(
        scenario, dirty, elevations.copy(), proj, with_units=True, with_sprites=True
    )
    with monkeypatch.context() as m:
        _zero_the_reaches(m)
        # with_sprites stays True: the knob under test is the REACHES, and
        # flipping the flag instead would measure a different thing entirely.
        plain = dirty_screen_bbox_iso(
            scenario, dirty, elevations.copy(), proj, with_units=True, with_sprites=True
        )
    assert widened is not None and plain is not None

    grew = (plain[0] - widened[0], plain[1] - widened[1], widened[2] - plain[2], widened[3] - plain[3])
    assert all(g > 0 for g in grew), (
        f"widened {widened} did not grow on all four sides over {plain} (grew by {grew}). "
        f"If a side is 0 the existing UNIT_FOOTPRINT_MAX_RADIUS dilation already "
        f"covered it, so this module is vacuous there -- check the fixture's sprite "
        f"canvas ({SPRITE_CANVAS}) and BASE_ELEVATION ({BASE_ELEVATION}) first."
    )

    for elevation in OBSERVED_ELEVATIONS:
        rect = _sprite_rect(scenario, proj, elevation)
        clamped = _clamped(rect, proj)
        assert clamped == rect, (
            f"the sprite rect {rect} at elevation {elevation} is clamped to {clamped}, "
            f"which makes the assertions below vacuous on the clamped side -- do not "
            f"shrink the map to save time"
        )
        assert not _covers(plain, rect), (
            f"the un-widened bbox {plain} already covers the sprite rect {rect} at "
            f"elevation {elevation}, so there is no defect here to catch"
        )
        assert _covers(widened, rect), (
            f"the widened bbox {widened} does not cover the sprite rect {rect} at "
            f"elevation {elevation} -- an edit here would leave a stale fragment"
        )


def test_zeroing_the_reaches_reproduces_the_sprites_off_bbox_exactly(sprite_install, monkeypatch):
    """Test 3. The mutation knob has to be honest.

    Every other test reverts the widening by zeroing the reaches. If that is
    only ASSUMED equivalent to having no widening at all, then a broken
    widening and a zeroed one fail identically and the red halves prove less
    than they look. Pin it: reaches at 0 must reproduce the with_sprites=False
    bbox exactly, which also catches the 1px rounding slack in
    _sprite_reach_px leaking into the result.

    This is also the test that goes red if the fixture ever regains an inert
    SPRITES_ENABLED monkeypatch and the with_sprites= arguments get dropped
    with it: both sides would then compute the sprites-off bbox and agree
    trivially. It only means something while the two sides genuinely differ.
    """
    scenario, dirty, elevations, proj = _edited_scenario()

    off = dirty_screen_bbox_iso(
        scenario, dirty, elevations.copy(), proj, with_units=True, with_sprites=False
    )
    with monkeypatch.context() as m:
        _zero_the_reaches(m)
        zeroed = dirty_screen_bbox_iso(
            scenario, dirty, elevations.copy(), proj, with_units=True, with_sprites=True
        )

    assert zeroed == off, (
        f"reaches at 0 gave {zeroed} but sprites-off gave {off} -- the mutation knob is "
        f"not a clean revert, so this module's red halves measure the knob, not the widening"
    )


@pytest.mark.parametrize(
    ("span", "coord"),
    [
        ((1, 1), (60.3, 60.7)),   # odd span, arbitrary float -- the non-building case
        ((2, 2), (60.0, 60.0)),   # even, aligned: ax == tile_x0, the LEFT bound
        ((2, 2), (60.5, 60.5)),   # even, off-parity: ax == tile_x0 + 2*half_w, the RIGHT bound
        ((2, 2), (60.0, 60.5)),   # mixed: ay == tile_origin_y + half_h -- needs the BOTTOM slack
        ((2, 2), (60.5, 60.0)),   # mixed: ay == tile_origin_y - half_h -- needs the TOP slack
        ((4, 1), (60.0, 60.3)),   # a gate: mixed span parity within one unit
        ((8, 8), (60.0, 60.0)),   # the widest span in BUILDING_TILE_SPANS
    ],
)
def test_the_bbox_covers_the_sprite_at_every_span_parity(sprite_install, monkeypatch, span, coord):
    """Test 4. The F2 check: the anchor is only pinned to its tile's diamond to
    within HALF A TILE, in each axis independently.

    _span_start has a half-tile branch, so `fx - int(x)` and `fy - int(y)` each
    range over {0, 0.5, 1} on their own. That puts ax anywhere in [tile_x0,
    tile_x0 + 2*half_w] and ay anywhere in [tile_origin_y +- half_h]. The naive
    "fy - fx == y - x" reading is wrong, and dropping the half_h term
    reproduces this exact bug class -- the two mixed-parity rows above clear
    the bottom and top edges by a single pixel with it and miss by half a tile
    without it.

    Off-parity placements are real, not theoretical: unit_tile_bounds' own
    docstring records 46 of 158 Mills and 32 of 254 Castles in the example
    corpus sitting on the parity their size does not expect.

    Asserts the FULL returned bbox, never _sprite_reach_px in isolation -- the
    padding is only correct in combination with the tile origin, the elevation
    sweep and the slack terms, and testing the part would let the whole drift.
    And it asserts the parity relation itself rather than assuming it: the
    duck-typed unit carries floats straight through _span_start, which is
    exactly where the naive claim breaks.
    """
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, CONST, span)
    scenario, dirty, elevations, proj = _edited_scenario(*coord)
    bbox = dirty_screen_bbox_iso(
        scenario, dirty, elevations.copy(), proj, with_units=True, with_sprites=True
    )
    assert bbox is not None

    for elevation in OBSERVED_ELEVATIONS:
        rect = _sprite_rect(scenario, proj, elevation)
        tile_x0, tile_y = iso_geometry.tile_screen_origin(EDIT_X, EDIT_Y, elevation, proj)
        # The anchor is the rect's own hotspot, recovered from the rect and the
        # known centre hotspot of the fixture's square sprite.
        ax, ay = rect[0] + (rect[2] - rect[0]) // 2, rect[1] + (rect[3] - rect[1]) // 2
        assert tile_x0 <= ax <= tile_x0 + 2 * proj.half_w, (
            f"ax {ax} escaped [tile_x0, tile_x0 + 2*half_w] = "
            f"[{tile_x0}, {tile_x0 + 2 * proj.half_w}] for span {span} at {coord}"
        )
        # The same statement as the x band above, and deliberately written the
        # same shape: the anchor lands inside the edit tile's own diamond
        # BOUNDING BOX. tile_screen_origin returns that box's top-left, so the
        # band is [tile_y, tile_y + 2*half_h] -- the tile's ground centre sits
        # at its midpoint, tile_y + half_h.
        #
        # This used to read `abs(ay - tile_y) <= half_h`, a band centred on the
        # box's TOP rather than its centre, which is precisely the half-tile
        # error that let every sprite render floating above its tile while this
        # test stayed green. The x half was always centre-relative; only y was
        # wrong, in the test and in the code alike.
        assert tile_y <= ay <= tile_y + 2 * proj.half_h, (
            f"ay {ay} escaped [tile_y, tile_y + 2*half_h] = "
            f"[{tile_y}, {tile_y + 2 * proj.half_h}] for span {span} at {coord}"
        )

        clamped = _clamped(rect, proj)
        assert clamped == rect, f"sprite rect {rect} clamps to {clamped} -- assertion would be vacuous"
        assert _covers(bbox, rect), (
            f"bbox {bbox} misses the sprite rect {rect} at elevation {elevation} for "
            f"span {span} at {coord}. A miss on the top or bottom edge alone means the "
            f"half_h slack (F2) is gone; a miss left or right means half_w"
        )


# --- Batch D's D3: extra_anchor_tiles ------------------------------------
#
# These three tests cannot drive UnitEditModel: this file's Unit is a
# duck-typed dataclass appended straight to unit_manager.units (see
# _fixture_scenario above), never tracked by UnitEditModel, and
# UnitEditModel.__init__ additionally requires units_write_supported, exactly
# 9 unit sections and a byte-verification of every unit this fixture cannot
# satisfy. So the move here is unit.x/unit.y assigned directly plus a hand
# `scenario.unit_gen += 1` -- the first consumer of D1's own documented
# direct-mutation contract (LoadedScenario.unit_gen's own docstring: a caller
# that mutates unit_manager.units without going through UnitEditModel must
# bump unit_gen itself, or a memo built before the mutation stays stale).


def _terrain_index(mm, x: int, y: int) -> int:
    return next(i for i, t in enumerate(mm.terrain) if (int(t.x), int(t.y)) == (x, y))


def test_a_unit_move_with_extra_anchor_tiles_covers_both_old_and_new_sprite_rects(
    sprite_install, monkeypatch
):
    """extra_anchor_tiles is what keeps a moved unit's OLD sprite fragment
    inside the bbox: anchor_tiles(scenario) is read live, i.e. POST-move, so
    it no longer contains the unit's old tile at all -- without this
    parameter the old fragment would never get padded in and would be left
    stale on screen."""
    scenario = _fixture_scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    old_x, old_y = EDIT_X, EDIT_Y
    new_x, new_y = EDIT_X + 1, EDIT_Y

    old_rects = {e: _clamped(_sprite_rect(scenario, proj, e), proj) for e in (BASE_ELEVATION,)}

    unit = scenario.unit_manager.units[1][0]
    unit.x, unit.y = float(new_x), float(new_y)
    scenario.unit_gen += 1

    new_rects = {e: _clamped(_sprite_rect(scenario, proj, e), proj) for e in (BASE_ELEVATION,)}

    dirty = [_terrain_index(mm, old_x, old_y), _terrain_index(mm, new_x, new_y)]
    bbox = dirty_screen_bbox_iso(
        scenario, dirty, elevations.copy(), proj, with_units=True, with_sprites=True,
        elevation_changed=set(), extra_anchor_tiles={(old_x, old_y)},
    )
    assert bbox is not None
    for e in (BASE_ELEVATION,):
        assert _covers(bbox, old_rects[e]), f"bbox {bbox} misses the OLD sprite rect {old_rects[e]} at {e}"
        assert _covers(bbox, new_rects[e]), f"bbox {bbox} misses the NEW sprite rect {new_rects[e]} at {e}"


def test_without_extra_anchor_tiles_the_same_move_misses_the_old_sprite_rect(
    sprite_install, monkeypatch
):
    """Non-vacuity, same shape test 2 uses: this is the test that fails if
    D5 ever stops passing the pre-edit footprint tiles into extra_anchor_tiles.

    Moves 40 tiles, not 1: the fixture's sprite reach (384px each side) is so
    large that a 1-tile move's own padding around the NEW anchor already
    reaches back and covers the OLD rect anyway, which would make this
    assertion fail for a reason that has nothing to do with the parameter
    under test. 40 tiles (1280px of horizontal screen separation alone) clears
    that by a wide margin while staying on this 120x120 map."""
    scenario = _fixture_scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    old_x, old_y = EDIT_X, EDIT_Y
    new_x, new_y = EDIT_X + 40, EDIT_Y

    old_rects = {e: _clamped(_sprite_rect(scenario, proj, e), proj) for e in (BASE_ELEVATION,)}

    unit = scenario.unit_manager.units[1][0]
    unit.x, unit.y = float(new_x), float(new_y)
    scenario.unit_gen += 1

    dirty = [_terrain_index(mm, old_x, old_y), _terrain_index(mm, new_x, new_y)]
    bbox = dirty_screen_bbox_iso(
        scenario, dirty, elevations.copy(), proj, with_units=True, with_sprites=True,
        elevation_changed=set(),
    )  # no extra_anchor_tiles
    assert bbox is not None
    for e in (BASE_ELEVATION,):
        assert not _covers(bbox, old_rects[e]), (
            f"bbox {bbox} already covers the OLD sprite rect {old_rects[e]} at {e} without "
            f"extra_anchor_tiles -- this test no longer proves the parameter is load-bearing"
        )


def test_extra_anchor_tiles_covers_a_multi_tile_old_footprint_not_just_the_anchor(
    sprite_install, monkeypatch
):
    """The gate-orientation case (set_unit_const swaps among spans (4, 1),
    (1, 4) and two 4x4 diagonals, so a gate's old footprint is not a subset
    of its new one), without dragging in gate_orientation/BUILDING_TILE_
    OFFSETS: BUILDING_TILE_SPANS[CONST] monkeypatched between the two
    _sprite_rect calls simulates the same defect -- a multi-tile OLD
    footprint that extra_anchor_tiles must carry in full, not just the
    single anchor tile a naive caller might pass instead."""
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, CONST, (4, 1))
    scenario = _fixture_scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    unit = scenario.unit_manager.units[1][0]

    old_footprint = set(render.unit_occupied_tiles(unit, mm.map_width, mm.map_height))
    assert len(old_footprint) > 1, "fixture must actually span more than one tile to prove anything"
    old_rects = {e: _clamped(_sprite_rect(scenario, proj, e), proj) for e in (BASE_ELEVATION,)}

    monkeypatch.setitem(render.BUILDING_TILE_SPANS, CONST, (1, 4))
    unit.x, unit.y = float(EDIT_X + 2), float(EDIT_Y)
    scenario.unit_gen += 1

    new_footprint = set(render.unit_occupied_tiles(unit, mm.map_width, mm.map_height))
    assert not old_footprint & new_footprint, "old and new footprints must not overlap, or this proves nothing"

    dirty_xy = old_footprint | new_footprint
    dirty = [i for i, t in enumerate(mm.terrain) if (int(t.x), int(t.y)) in dirty_xy]

    bbox = dirty_screen_bbox_iso(
        scenario, dirty, elevations.copy(), proj, with_units=True, with_sprites=True,
        elevation_changed=set(), extra_anchor_tiles=old_footprint,
    )
    assert bbox is not None
    for e, rect in old_rects.items():
        assert _covers(bbox, rect), (
            f"bbox {bbox} misses the OLD (4, 1)-span sprite rect {rect} at {e} -- "
            f"extra_anchor_tiles must carry the WHOLE old footprint, not just one tile"
        )


# --- the oracle ---------------------------------------------------------


def test_patch_after_edit_matches_a_fresh_full_render(sprite_install, monkeypatch):
    """Test 1. The whole-canvas oracle, in the shape
    tests/test_sloped_edit.py::test_patch_after_edit_matches_a_fresh_full_render
    and tools/verify_iso_chunks.py's check_patch_after_scripted_ops already use:
    warm every chunk, edit, bbox, patch, and demand the cached pixels be
    indistinguishable from rendering the post-edit scenario from scratch.

    This is the assertion that fails for a wrong pad, a dropped slack term and a
    missing elevation sweep alike; the cheap tests above exist to say WHICH.

    The second half is the mutation check, reusing the one full render: the same
    sequence with the reaches zeroed must NOT match. Without it a byte-identical
    first half could just mean the sprite happened to sit inside the existing
    dilation. Both halves stay in the default tier -- about 3s total, the same
    order tests/test_sloped_edit.py already pays there -- because a guard that
    only runs behind a marker is a guard that does not run.
    """
    scenario, dirty, elevations, proj = _edited_scenario()
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)

    cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk, PRE-edit

    bbox = dirty_screen_bbox_iso(
        scenario, dirty, cache.elevations, proj, with_units=True, with_sprites=True
    )
    assert bbox is not None, "a legal in-range edit must not decline the incremental path"
    cache.patch(bbox)

    full, _elev, _proj = render_terrain_iso_with_proj(scenario, with_sprites=True)
    assert np.array_equal(cache.render_rect(0, 0, canvas_w, canvas_h), full[:canvas_h, :canvas_w])

    # Mutation: same sequence, widening neutered. A second scenario rather than
    # reusing the one above, whose tiles are already post-edit; `full` is reused
    # because the two scenarios are identical once both edits have landed.
    _zero_the_reaches(monkeypatch)
    scenario2, dirty2, elevations2, proj2 = _edited_scenario()
    # sprites=True on BOTH caches. It is easy to read this second one as not
    # needing it -- it is the neutered arm -- but the knob being neutered is
    # the REACHES, not the sprites. With sprites off here the canvas would
    # differ from `full` everywhere a sprite belongs, so the `assert not
    # array_equal` below would pass no matter what the reaches did: green
    # while measuring nothing, in the one assertion that exists to prove this
    # oracle is not vacuous.
    cache2 = IsoChunkCache(scenario2, elevations2, proj2, tile_px, sprites=True)
    cache2.render_rect(0, 0, canvas_w, canvas_h)
    bbox2 = dirty_screen_bbox_iso(
        scenario2, dirty2, cache2.elevations, proj2, with_units=True, with_sprites=True
    )
    assert bbox2 is not None
    cache2.patch(bbox2)

    assert not np.array_equal(
        cache2.render_rect(0, 0, canvas_w, canvas_h), full[:canvas_h, :canvas_w]
    ), (
        "with the sprite reaches zeroed the patch STILL matched a fresh full render, so "
        "the widening is not what makes the first half pass -- the sprite is sitting "
        "inside the pre-existing dilation and this oracle is vacuous"
    )


def _warm_both_levels(cache, probe):
    """Warms `probe` (a level-0 pixel rect) at level 0 AND at COARSER_LEVEL,
    and returns the (mip, cx, cy) keys that must still be resident afterwards.

    A SUB-RECT, not the whole canvas, and that is the whole trick. The byte
    budget _init_max_chunks sets is exactly one level-0 canvas
    (canvas_dims() area * 3), by design: "N levels redistribute memory, they
    don't multiply it". So warming the full canvas at level 0 and then at a
    coarser level evicts level-0 chunks to pay for the coarse ones (MEASURED
    on this fixture: 120 level-0 chunks in, 90 left after the second warm).
    Evicted chunks are exactly the ones patch() skips and get_chunk()
    recomposites fresh against the post-edit state, so a whole-canvas
    two-level warm produces an oracle that cannot fail. It was written that
    way first and passed against a deliberately neutered widening.
    """
    lvl_probe = cache._bbox_to_level(COARSER_LEVEL, probe)
    cache.render_rect(*probe, mip=0)
    cache.render_rect(*lvl_probe, mip=COARSER_LEVEL)
    keys = set()
    for mip, rect in ((0, probe), (COARSER_LEVEL, lvl_probe)):
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, *rect)
        keys |= {(mip, cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)}
    return lvl_probe, keys


def test_patch_after_edit_matches_a_fresh_full_render_at_a_coarser_mip(sprite_install, monkeypatch):
    """The same oracle, with a COARSER mip level resident as well as level 0.

    Every other test in this module runs at level 0 alone, because warming via
    render_rect(0, 0, canvas_w, canvas_h) leaves only the reference level
    resident and _ChunkCacheBase.patch iterates resident levels only. So the
    widening had never actually been patched at a mip.

    The widening is believed correct here, and that belief is the reason to
    measure it: _bbox_to_level floors the low edge and ceils the high one, so a
    level-0 rect covering the sprite converts to a superset at any coarser
    level, and the sprite shrinks by the same exact ratio (is_exact_mip). The
    argument is sound. It was still an argument standing in for a measurement,
    which is what the rest of this module deliberately avoids.

    ONE dirty_screen_bbox_iso call and ONE patch(), asserted at both levels.
    That is the contract: the caller computes the dirty rect once, in level-0
    screen space, and every resident level has to come out right.

    Scoped to the sprite's own neighbourhood rather than the whole canvas, for
    the eviction reason _warm_both_levels() documents, and compared against the
    matching crop of a full render at each level.
    """
    scenario, dirty, elevations, proj = _edited_scenario()
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)

    cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)
    # An assert, not a skip: this fixture is the shipped 120x120 template at
    # fixed settings, so the level either enumerates here always or never. A
    # skip would let the whole test quietly stop running.
    assert COARSER_LEVEL in cache.mip_levels(), (
        f"level {COARSER_LEVEL} not enumerated on the fixture (levels={cache.mip_levels()})"
    )
    probe = _clamped(_sprite_rect(scenario, proj, EDIT_ELEVATION), proj)
    lvl_probe, warmed = _warm_both_levels(cache, probe)
    # The guard that stops this test going vacuous the way its first draft did.
    # Both levels' probe chunks must survive the second warm; a chunk that was
    # evicted is recomposited fresh below and could not fail whatever the
    # widening did.
    missing = sorted(k for k in warmed if not cache.has_chunk(*k))
    assert not missing, f"probe chunks evicted before patch(), oracle would be vacuous: {missing}"

    bbox = dirty_screen_bbox_iso(
        scenario, dirty, cache.elevations, proj, with_units=True, with_sprites=True
    )
    assert bbox is not None, "a legal in-range edit must not decline the incremental path"
    cache.patch(bbox)

    full, _elev, _proj = render_terrain_iso_with_proj(scenario, with_sprites=True)
    x0, y0, x1, y1 = probe
    assert np.array_equal(cache.render_rect(*probe, mip=0), full[y0:y1, x0:x1]), (
        "level 0 diverged after a patch() that also had a coarser level resident"
    )

    full_lvl = _full_render_at_level(cache, monkeypatch, scenario)
    lx0, ly0, lx1, ly1 = lvl_probe
    assert np.array_equal(
        cache.render_rect(*lvl_probe, mip=COARSER_LEVEL), full_lvl[ly0:ly1, lx0:lx1]
    ), f"level {COARSER_LEVEL} diverged after patch()"

    # Mutation, at the coarse level only, since the oracle above already
    # covers level 0's. full_lvl is reused: the reaches are read by
    # dirty_screen_bbox_iso alone, never by a full render, so a fresh full
    # render of the identical post-edit scenario would be byte-identical.
    _zero_the_reaches(monkeypatch)
    scenario2, dirty2, elevations2, proj2 = _edited_scenario()
    # sprites=True on this cache too. It is easy to read the neutered arm as
    # not needing it, but the knob being neutered is the REACHES, not the
    # sprites; with sprites off the canvas would differ from full_lvl
    # everywhere a sprite belongs and the assertion below would pass no matter
    # what the reaches did.
    cache2 = IsoChunkCache(scenario2, elevations2, proj2, tile_px, sprites=True)
    _warm_both_levels(cache2, probe)
    bbox2 = dirty_screen_bbox_iso(
        scenario2, dirty2, cache2.elevations, proj2, with_units=True, with_sprites=True
    )
    assert bbox2 is not None
    cache2.patch(bbox2)

    assert not np.array_equal(
        cache2.render_rect(*lvl_probe, mip=COARSER_LEVEL), full_lvl[ly0:ly1, lx0:lx1]
    ), (
        f"with the sprite reaches zeroed the level-{COARSER_LEVEL} patch STILL matched a fresh "
        "full render, so the widening is not what makes the first half pass and this check is "
        "vacuous at a coarser mip"
    )


# --- the constants, against real assets ---------------------------------


@pytest.mark.corpus
def test_no_real_sprite_reaches_past_its_committed_constant():
    """The only check that can keep unit_sprites.MAX_SPRITE_REACH_* honest.

    Everything above runs on synthetic bytes with those constants monkeypatched
    to match, so the whole default tier would stay green if a game patch shipped
    a sprite taller than b_west_wonder_britons_x1 -- and the symptom would be a
    rare stale fragment on one building, which is exactly the kind of defect
    that gets misattributed for months.

    Marked `corpus` for the tier's real meaning, "needs untracked local data",
    though the data here is an AoE2:DE install rather than examples/ -- so it
    skips on the install, not on the scenario corpus. Reuses
    tools/scan_sprite_reach.py's own scan() rather than re-deriving the walk:
    a second implementation could drift from the one whose output produced the
    committed numbers, which would make this test agree with nothing.

    **Needs AOE2DE_INSTALL_PATH, not the configured install**, and that is not
    an oversight to "fix" by reaching around it. conftest._isolated_settings is
    autouse and redirects CONFIG_PATH at every test, so a configured install is
    deliberately invisible to the whole suite; the env var is the one route
    asset_source.get_install_path() honours above it. That keeps reading real
    game assets an explicit opt-in rather than something a test does to whoever
    happens to have the game installed.
    """
    install = asset_source.get_install_path()
    if install is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The "
            "install configured in config.yaml does NOT count here: "
            "conftest._isolated_settings redirects CONFIG_PATH for every test, so "
            "the env var is the only route in (see this test's docstring)"
        )
    scan = conftest.load_verify_module("scan_sprite_reach")

    graphics_dir = Path(install) / unit_sprites.GRAPHICS_SUBPATH
    # Mirrors tools/scan_sprite_reach.py's own main() -- composite pieces are
    # scanned at their own (dx, dy) offset from the shared unit anchor, not
    # at (0, 0), same reasoning as that module's docstring.
    entries: set[tuple[str, int, int]] = set()
    for gm_entry in unit_sprites.graphic_map().values():
        entries.add((gm_entry["file_name"], 0, 0))
        for piece in gm_entry.get("pieces", ()):
            entries.add((piece["file_name"], piece["dx"], piece["dy"]))
    entries = sorted(entries)
    maxima, worst, readable, unreadable = scan.scan(graphics_dir, entries)

    assert readable > 0, (
        f"{unreadable} of {len(entries)} referenced (file, offset) entries were readable "
        f"at {graphics_dir} -- the install is configured but has no sprite data, so this "
        f"check proves nothing rather than passing"
    )
    for i, (label, const) in enumerate(scan.DIRECTIONS):
        committed = getattr(unit_sprites, const)
        assert maxima[i] <= committed, (
            f"{worst[i]} reaches {maxima[i]}px {label} of its hotspot, past the "
            f"committed {const} = {committed}. An edit beside it would leave a stale "
            f"sprite fragment. Re-run tools/scan_sprite_reach.py and update the four "
            f"constants in descape/unit_sprites.py together"
        )
