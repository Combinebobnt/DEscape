"""Large-elevation-delta coverage for `_dirty_screen_bbox` (bbox-elev-sweep plan, Step 2).

The oracle is a DIFF MASK, not a patch-vs-full comparison alone: render a
window around the edit before and after from fresh state, and require every
pixel that changed to sit inside the bbox. A bbox far bigger than the change
passes a byte-identity check vacuously; the mask says exactly which rows
escaped when one doesn't.

Each case also patches a warm cache and compares it with a fresh one, the
check a stale fragment actually shows up in.

Edits are written straight into `tile.elevation` (no propagation) so a single
tile jumps the whole legal range next to untouched neighbours. That is the
worst case for a skirt: a lowered tile makes its higher neighbours hang a
15-level skirt down to it, which none of those neighbours' own elevations
predict. The propagated case covers the realistic cone a real click makes.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sprite_edit_bbox import CONST, Unit, sprite_install  # noqa: F401 -- fixture

from descape import iso_geometry, render
from descape.elevation_tools import set_tile_elevation
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_palette import BUILDING_TILE_SPANS

EDIT = (60, 60)
BIG_BUILDING_CONST = 182
assert BUILDING_TILE_SPANS[BIG_BUILDING_CONST] == (5, 5)
# Window radius in tiles: the 5x5 building's far footprint tiles are 2 away,
# plus the skirt/Sloped corner ring, plus margin. The propagated cone spreads
# one tile per level, so it gets PROPAGATED_TARGET more.
WINDOW_RADIUS = 5
PROPAGATED_TARGET = 5
MIN_E, MAX_E = iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION


def _scenario(base: int, units: str):
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = base
    ex, ey = EDIT
    players = scenario.unit_manager.units
    if units == "back_diagonal":
        # Up-screen of the edit, sharing one corner with it: a Sloped sprite
        # here rises with the edit while its own tile never changes.
        players[2].append(Unit(ex + 1.5, ey - 0.5, CONST))
    elif units == "on_edit":
        # Own tile on the edited tile, footprint reaching 2 tiles past it.
        players[1].append(Unit(ex + 0.5, ey + 0.5, BIG_BUILDING_CONST))
        # A sprite-bearing unit on the edited tile and one beside it (Sloped's ring).
        players[2].append(Unit(ex + 0.5, ey + 0.5, CONST))
        players[2].append(Unit(ex + 1.5, ey + 0.5, CONST))
    return scenario


def _raise(mm):
    mm.get_tile(*EDIT).elevation = MAX_E


def _lower(mm):
    """A 2x2 block, not one tile: under Sloped's "max" corner rule a lone
    lowered tile keeps all four corners at its neighbours' height and paints
    nothing different. The block's shared inner corner is what drops."""
    ex, ey = EDIT
    for dx in (0, 1):
        for dy in (0, 1):
            mm.get_tile(ex + dx, ey + dy).elevation = MIN_E


def _paint_cliff_top(mm):
    mm.get_tile(*EDIT).terrain_id = 5


def _propagated(mm):
    set_tile_elevation(mm, *EDIT, PROPAGATED_TARGET)


def _cliff_below_edit(scenario):
    """The edited tile's skirt neighbours sit at the bottom of the range, so a
    texture change on it repaints a skirt its own elevation never reaches."""
    ex, ey = EDIT
    mm = scenario.map_manager
    mm.get_tile(ex - 1, ey).elevation = MIN_E
    mm.get_tile(ex, ey + 1).elevation = MIN_E


# (id, base elevation, pre-edit setup, edit, units, window radius). The paint
# case has no units: the building's own marker would cover the tile.
EDITS = [
    ("raise_0_to_15", MIN_E, None, _raise, "on_edit", WINDOW_RADIUS),
    ("lower_15_to_0", MAX_E, None, _lower, "on_edit", WINDOW_RADIUS),
    ("paint_cliff_top", MAX_E, _cliff_below_edit, _paint_cliff_top, "none", WINDOW_RADIUS),
    ("propagated_raise", MIN_E, None, _propagated, "on_edit", WINDOW_RADIUS + PROPAGATED_TARGET),
    ("raise_under_back_diagonal_sprite", MIN_E, None, _raise, "back_diagonal", WINDOW_RADIUS),
]


def _make_cache(style, scenario, sprites):
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = render.elevations_and_proj(scenario)
        return IsoChunkCache(scenario, elevations, proj, tile_px, sprites=sprites), elevations, proj
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
    return SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=sprites), elevations, proj


def _window(proj, canvas_dims, radius):
    ex, ey = EDIT
    pad_l, pad_u, pad_r, pad_d = render._sprite_reach_px(proj)
    x0 = y0 = x1 = y1 = None
    for x in range(ex - radius, ex + radius + 1):
        for y in range(ey - radius, ey + radius + 1):
            tx0, ty0, tx1, ty1 = iso_geometry.tile_screen_bounds_swept(x, y, proj)
            x0 = tx0 if x0 is None else min(x0, tx0)
            y0 = ty0 if y0 is None else min(y0, ty0)
            x1 = tx1 if x1 is None else max(x1, tx1)
            y1 = ty1 if y1 is None else max(y1, ty1)
    cw, ch = canvas_dims
    return max(0, x0 - pad_l), max(0, y0 - pad_u), min(cw, x1 + pad_r), min(ch, y1 + pad_d)


def _dirty(mm, before_state):
    return [
        i for i, t in enumerate(mm.terrain)
        if (int(t.elevation), int(t.terrain_id)) != before_state[i]
    ]


@pytest.mark.parametrize("style", ["stepped", "sloped"])
@pytest.mark.parametrize("sprites", [False, True], ids=["sprites_off", "sprites_on"])
@pytest.mark.parametrize("edit_id, base, setup, edit, units, radius", EDITS, ids=[e[0] for e in EDITS])
def test_changed_pixels_stay_inside_the_bbox(request, style, sprites, edit_id, base, setup, edit, units, radius):
    if sprites:
        request.getfixturevalue("sprite_install")
    scenario = _scenario(base, units)
    if setup is not None:
        setup(scenario)
    mm = scenario.map_manager

    cache, elevations, proj = _make_cache(style, scenario, sprites)
    wx0, wy0, wx1, wy1 = _window(proj, cache.canvas_dims(), radius)
    before = cache.render_rect(wx0, wy0, wx1, wy1).copy()

    before_state = [(int(t.elevation), int(t.terrain_id)) for t in mm.terrain]
    edit(mm)
    dirty = _dirty(mm, before_state)
    assert dirty, "fixture edit changed nothing"

    bbox_fn = render.dirty_screen_bbox_iso if style == "stepped" else render.dirty_screen_bbox_sloped
    elevation_changed: set = set()
    bbox = bbox_fn(
        scenario, dirty, elevations, proj, with_units=True, with_sprites=sprites,
        elevation_changed=elevation_changed,
    )
    assert bbox is not None
    cache.patch(bbox, elevation_changed)
    patched = cache.render_rect(wx0, wy0, wx1, wy1)

    fresh_cache, _elev, _proj = _make_cache(style, scenario, sprites)
    fresh = fresh_cache.render_rect(wx0, wy0, wx1, wy1)

    changed = (before != fresh).any(axis=2)
    ys, xs = np.nonzero(changed)
    assert ys.size, "the edit changed no pixels in the window -- fixture is vacuous"
    dx0, dy0 = int(xs.min()) + wx0, int(ys.min()) + wy0
    dx1, dy1 = int(xs.max()) + wx0 + 1, int(ys.max()) + wy0 + 1
    # The window must not be what bounds the diff, or the check is blind past it.
    cw, ch = cache.canvas_dims()
    assert (dx0 > wx0 or wx0 == 0) and (dy0 > wy0 or wy0 == 0) and (dx1 < wx1 or wx1 == cw) and (
        dy1 < wy1 or wy1 == ch
    ), f"diff {dx0, dy0, dx1, dy1} touches window {wx0, wy0, wx1, wy1}; widen the window radius"
    bx0, by0, bx1, by1 = bbox
    assert bx0 <= dx0 and by0 <= dy0 and bx1 >= dx1 and by1 >= dy1, (
        f"changed pixels {dx0, dy0, dx1, dy1} escape bbox {bbox}"
    )
    assert np.array_equal(patched, fresh), "patched cache differs from a fresh render"
