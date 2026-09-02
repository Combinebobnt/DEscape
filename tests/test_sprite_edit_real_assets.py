"""P3-g5's edit-path widening, against REAL game sprites in a REAL running
window -- the check that had no way to run until P3-g's `Show sprites` toggle
existed.

Everything in tests/test_sprite_edit_bbox.py runs on synthetic `.sld` bytes
with the four `MAX_SPRITE_REACH_*` constants monkeypatched to match the
fixture, so fixture and widening agree by construction. That is the right
trade for the default tier, but it means the widening has never once been
watched cover an actual game asset, and never at all through the viewer's own
edit path. This module closes both gaps.

**Assertion, not inspection.** The temptation with a "visual check" is to
capture a crop and look at it. A crop framed by an approximated projection
frames the wrong tiles, and then "looks fine" gets said about an image that
never contained the unit. So this compares the patched canvas against a fresh
full render of the post-edit scenario and demands byte equality -- the same
oracle tools/verify_iso_chunks.py and tests/test_sloped_edit.py already use.

**Needs AOE2DE_INSTALL_PATH, not the configured install**, for exactly the
reason test_sprite_edit_bbox.py's own corpus test spells out: conftest's
autouse _isolated_settings redirects CONFIG_PATH at every test, so a
configured install is deliberately invisible to the suite, and the env var is
the one route asset_source.get_install_path() honours above it.
"""

from __future__ import annotations

import numpy as np
import pytest

import conftest
from descape import asset_source, render, unit_sprites
from descape.elevation_tools import set_tile_elevation
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# The Britons Wonder. Chosen because tools/scan_sprite_reach.py resolves
# b_west_wonder_britons_x1 as the asset that sets MAX_SPRITE_REACH_UP (650px)
# -- i.e. the single sprite that most out-reaches the terrain-only dilation
# this widening exists to correct. A small unit would leave the whole module
# passing on the pre-existing UNIT_FOOTPRINT_MAX_RADIUS slack.
WONDER_CONST = 276

# Elevation 14 -> 15, matching tests/test_sprite_edit_bbox.py and for the same
# non-obvious reason: the swept bbox carries (max_elev - e) * elev_step of free
# vertical slack, so a sprite low on the map is covered for the wrong reason.
# At max_elev that slack is zero and the widening is the only thing holding the
# bbox open.
BASE_ELEVATION = 14
EDIT_ELEVATION = 15
ANCHOR_X = ANCHOR_Y = 60

# The 25% stop, where the widening does the most work: a shorter elevation step
# shrinks the free slack described above.
ELEV_STEP_PCT = 25


def _require_install():
    install = asset_source.get_install_path()
    if install is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )
    if WONDER_CONST not in unit_sprites.graphic_map():
        pytest.skip(f"unit_const {WONDER_CONST} is not in this install's graphic map")
    return install


class _Unit:
    """The four attributes the sprite/footprint paths actually read, duck-typed
    -- the same posture tests/test_sprite_chunks.py and test_sprite_edit_bbox.py
    already take rather than building a real genieutils unit."""

    def __init__(self, x: float, y: float, unit_const: int) -> None:
        self.x, self.y, self.unit_const, self.rotation = x, y, unit_const, 0.0


def _place_wonder(scenario) -> None:
    """Flattens the map and drops one Wonder on it, in place.

    Placed programmatically on the blank template rather than hunting an
    examples/ file that happens to contain a Wonder: the plan called this the
    better option and it is, because the tile coordinates and the elevation
    become controlled inputs instead of discovered ones. Player 1, so the tint
    cannot be confused with terrain.
    """
    for tile in scenario.map_manager.terrain:
        tile.elevation = BASE_ELEVATION
    scenario.unit_manager.units[1].append(_Unit(float(ANCHOR_X), float(ANCHOR_Y), WONDER_CONST))


def _edit_and_dirty(mm) -> list[int]:
    """One real elevation edit at the Wonder's anchor tile, returning the
    terrain INDICES it actually changed -- one click propagates through
    _elevation_tile_recursion to many tiles, and the propagated set is what a
    stroke hands _apply_dirty."""
    before = [int(tile.elevation) for tile in mm.terrain]
    set_tile_elevation(mm, ANCHOR_X, ANCHOR_Y, EDIT_ELEVATION)
    changed = [i for i, tile in enumerate(mm.terrain) if int(tile.elevation) != before[i]]
    assert changed, "set_tile_elevation produced no change -- fixture is broken"
    return changed


def test_a_stroke_beside_a_real_wonder_repaints_its_whole_sprite_in_a_live_window():
    """The one P3-g5 could never run: a real window, real assets, the toggle,
    and the viewer's own edit path end to end.

    Deliberately drives ViewerWindow._apply_dirty rather than calling
    dirty_screen_bbox_iso directly. That is the whole point -- the widening's
    correctness now depends on _apply_dirty passing the cache's own
    sprites_enabled through, and this is the only test that walks that wiring
    with real sprite bytes behind it.
    """
    _require_install()
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH, elev_step_pct=ELEV_STEP_PCT)
    try:
        assert window.show_sprites_action.isEnabled(), (
            "the toggle should be reachable with an install configured and Stepped active"
        )
        _place_wonder(window.scenario)
        window._render_current()  # rebuild the cache over the placed Wonder

        window.show_sprites_action.setChecked(True)
        assert window._cache.sprites_enabled is True

        # Non-vacuity, checked BEFORE the edit: if the Wonder's sprite failed to
        # resolve (a renamed asset, a graphic-map miss), everything below would
        # compare two identical mark-only renders and pass while testing nothing.
        sprites = window._cache._level(0).sprites
        assert sprites is not None and sprites.by_anchor, (
            "no sprite layer built -- the Wonder's .sld did not resolve, so this "
            "oracle would be comparing two coloured-mark renders"
        )

        canvas_w, canvas_h = window._cache.canvas_dims(0)
        window._cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk, PRE-edit

        mm = window.scenario.map_manager
        window._apply_dirty(window.edit_history.apply("elevation", mm.terrain, lambda: _edit_and_dirty(mm)))

        full, _elev, _proj = render.render_terrain_iso_with_proj(window.scenario, with_sprites=True)
        assert np.array_equal(
            window._cache.render_rect(0, 0, canvas_w, canvas_h), full[:canvas_h, :canvas_w]
        ), (
            "the patched canvas differs from a fresh full render of the post-edit "
            "scenario -- an elevation stroke beside the Wonder left a stale sprite "
            "fragment, which is exactly the defect P3-g5's widening exists to prevent"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_real_wonder_sprite_actually_out_reaches_the_plain_dilation():
    """The mutation half, and the reason the test above means anything.

    A byte-identical patch could just mean the Wonder happens to sit inside the
    pre-existing UNIT_FOOTPRINT_MAX_RADIUS dilation, in which case the widening
    is doing nothing and the oracle above is vacuous against real assets even
    though it is not against the synthetic fixture. Zero the four reaches and
    the same comparison must FAIL.

    Runs at cache level rather than through a second window: nothing about the
    mutation involves the viewer wiring, and a window costs a full re-render.
    """
    _require_install()
    from descape.render import dirty_screen_bbox_iso
    from descape.render_cache import IsoChunkCache
    from descape.scenario_io import load_map_and_units

    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    _place_wonder(scenario)
    elevations, proj = render.elevations_and_proj(scenario)
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)

    cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)
    canvas_w, canvas_h = cache.canvas_dims(0)
    cache.render_rect(0, 0, canvas_w, canvas_h)

    with pytest.MonkeyPatch.context() as m:
        for name in (
            "MAX_SPRITE_REACH_LEFT",
            "MAX_SPRITE_REACH_UP",
            "MAX_SPRITE_REACH_RIGHT",
            "MAX_SPRITE_REACH_DOWN",
        ):
            m.setattr(unit_sprites, name, 0)
        dirty = _edit_and_dirty(mm)
        bbox = dirty_screen_bbox_iso(
            scenario, dirty, cache.elevations, proj, with_units=True, with_sprites=True
        )
        assert bbox is not None
        cache.patch(bbox)

    full, _elev, _proj = render.render_terrain_iso_with_proj(scenario, with_sprites=True)
    assert not np.array_equal(cache.render_rect(0, 0, canvas_w, canvas_h), full[:canvas_h, :canvas_w]), (
        "with the sprite reaches zeroed the patch STILL matched a fresh full render, so the "
        "widening is not what makes the live-window oracle pass -- the real Wonder sprite is "
        "sitting inside the pre-existing dilation and that test is vacuous against real assets"
    )
