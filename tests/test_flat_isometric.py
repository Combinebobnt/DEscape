"""Flat + Isometric View becomes a real isometric render (Flat+Isometric plan,
2026-09-03). Closes the follow-up P3-g7's footprint icons raised: squashed
under Flat's plain QTransform, a unit read as a lozenge lying on the ground
rather than a figure standing on it.

Ticking Isometric View in Flat now switches the RENDER PATH (a real
IsoChunkCache built on an all-zero elevation array), not the view transform --
see ViewerWindow._render_style's own docstring for the self._terrain_style
(what the user picked) vs self._render_style (what got built) split this
whole feature turns on.

Tests are ordered so the non-vacuity checks (real elevations, real sprites)
come before the oracles that would otherwise pass for the wrong reason -- the
same discipline tests/test_sprite_edit_bbox.py documents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import conftest
from descape import asset_source, iso_geometry, unit_pick, unit_sprites
from descape.elevation_tools import set_tile_elevation
from descape.render_cache import FlatChunkCache, IsoChunkCache
from test_unit_sprites import CONST, FILE_NAME, build_sld

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# Real, non-zero, non-default so a bug that left elevations untouched (rather
# than genuinely zeroed) cannot pass by coincidence -- same reasoning
# tests/test_sprite_edit_bbox.py's BASE_ELEVATION comment gives.
REAL_ELEVATION = 7
UNIT_X = UNIT_Y = 60


@dataclass
class Unit:
    """Duck-typed unit, matching tests/test_sprite_edit_bbox.py's own Unit,
    plus reference_id -- build_index()'s own unit_key() needs it, which that
    module's fixture never calls."""

    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    reference_id: int = 1


def _raise_all_tiles(window) -> None:
    """Real non-zero elevation everywhere, set directly on the loaded
    scenario's terrain -- mirrors _fixture_scenario() in
    tests/test_sprite_edit_bbox.py, just applied post-load since blank_window()
    already parsed the (flat) template."""
    for tile in window.scenario.map_manager.terrain:
        tile.elevation = REAL_ELEVATION


def _flat_iso_window(checked: bool = True):
    """blank_window() switched to Flat, with Isometric View forced to
    `checked` -- MapView._isometric defaults True (see map_view.py), so a
    freshly-opened Flat view starts in the real-iso state without any
    explicit toggle; tests that want the OFF state must pass checked=False."""
    window = conftest.blank_window()
    _raise_all_tiles(window)
    window.terrain_style_combo.setCurrentText("Flat")
    window.iso_action.setChecked(checked)
    return window


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """Copied from tests/test_sprite_toggle_viewer.py's own fixture of the
    same name -- a tmp install holding one small synthetic sprite, just
    enough for asset_source.is_available() to be True and CONST to resolve."""
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


def test_flat_iso_builds_a_stepped_cache_with_zeroed_elevations() -> None:
    window = _flat_iso_window()
    try:
        mm = window.scenario.map_manager
        real_elevations = {int(mm.get_tile(x, y).elevation) for x in range(mm.map_width) for y in range(mm.map_height)}
        assert real_elevations == {REAL_ELEVATION}, "fixture assumption: the scenario must carry real elevation first"

        assert isinstance(window._cache, IsoChunkCache), "Flat + Isometric View must build a real IsoChunkCache"
        assert window._iso_elevations is not None
        assert np.all(window._iso_elevations == 0), (
            "Flat + Isometric View must render at elevation 0 regardless of the scenario's real elevations"
        )
    finally:
        conftest.close_window(window)


def test_the_array_stays_zero_through_a_terrain_and_an_elevation_edit() -> None:
    """F2's own oracle, and the whole reason Step 1 (flatten_elevations)
    exists: without it, an edited tile re-raises itself to its real
    elevation on the next dirty-bbox call while every other tile stays flat,
    corrupting the shared array (and, with it, hit-testing) silently."""
    window = _flat_iso_window()
    try:
        mm = window.scenario.map_manager

        tile = mm.get_tile(UNIT_X, UNIT_Y)
        new_terrain_id = 0 if tile.terrain_id != 0 else 1
        tile.terrain_id = new_terrain_id
        dirty = [i for i, t in enumerate(mm.terrain) if (t.x, t.y) == (UNIT_X, UNIT_Y)]
        assert dirty, "fixture assumption: the terrain edit must actually change something"
        window._apply_dirty(dirty)
        assert np.all(window._iso_elevations == 0), "a terrain-only edit must not un-zero the array"

        elev_before = [int(t.elevation) for t in mm.terrain]
        set_tile_elevation(mm, UNIT_X, UNIT_Y, REAL_ELEVATION + 1 if REAL_ELEVATION < 15 else REAL_ELEVATION - 1)
        dirty = [i for i, t in enumerate(mm.terrain) if int(t.elevation) != elev_before[i]]
        assert dirty, "fixture assumption: the elevation edit must actually change something"
        window._apply_dirty(dirty)
        assert np.all(window._iso_elevations == 0), (
            "an elevation edit must not re-raise the edited tile to its real height -- this is the "
            "exact corruption F2 documents"
        )
    finally:
        conftest.close_window(window)


def test_flat_iso_off_still_builds_flat_chunk_cache_with_icons(sprite_install) -> None:
    window = _flat_iso_window(checked=False)
    try:
        assert isinstance(window._cache, FlatChunkCache)
        assert window._iso_elevations is None, "the un-transformed Flat view must never carry an elevation snapshot"
        assert window._cache._level_icons(0) is not None, "P3-g7's icons must still resolve, unchanged"
    finally:
        conftest.close_window(window)


def test_flat_iso_sprites_are_real_stepped_sprites(sprite_install) -> None:
    window = _flat_iso_window()
    try:
        window.scenario.unit_manager.units[1].append(Unit(UNIT_X, UNIT_Y, CONST))
        window._render_current(reset_view=False)  # rebuild the cache with the unit now on the scenario
        lvl = window._cache._level(0)
        assert lvl.sprites is not None, "Flat + Isometric View must build a real sprite layer, not icons"
        assert lvl.sprites.skip_ids, "at least one unit must have resolved to a real sprite"
    finally:
        conftest.close_window(window)


def test_set_isometric_does_not_apply_the_qtransform_in_flat_iso() -> None:
    """The mechanism Step 3 leans on: set_source() ends by calling
    set_isometric() again, and MapView._terrain_style is "stepped" for this
    mode (F1), so it takes MapView's own early return and skips the
    squash+rotate. A rotated transform has non-zero off-diagonal (shear)
    terms; a plain fit-to-view scale does not."""
    window = _flat_iso_window()
    try:
        t = window.map_view.transform()
        assert t.m12() == 0 and t.m21() == 0, "the QTransform must not be rotated in Flat + Isometric View"
    finally:
        conftest.close_window(window)


def test_toggling_with_no_scenario_open_does_not_raise() -> None:
    window = conftest.blank_window(load=False)
    try:
        window.iso_action.setChecked(True)
        window.iso_action.setChecked(False)
    finally:
        conftest.close_window(window)


def test_toggling_while_busy_does_not_reenter() -> None:
    window = _flat_iso_window(checked=False)
    try:
        before = window._cache
        window._busy = True
        try:
            window.iso_action.setChecked(True)
        finally:
            window._busy = False
        assert window._cache is before, "a toggle received while a render is in progress must not re-enter"
    finally:
        conftest.close_window(window)


def test_status_bar_reads_flat_and_iso_action_enabled_only_in_flat() -> None:
    window = _flat_iso_window()
    try:
        assert window.iso_action.isEnabled()
        assert "Flat" in window.mode_status_label.text(), (
            "the status bar must keep reading the user-facing style, not the render style"
        )

        window.terrain_style_combo.setCurrentText("Stepped")
        assert not window.iso_action.isEnabled()
        window.terrain_style_combo.setCurrentText("Sloped")
        assert not window.iso_action.isEnabled()
    finally:
        conftest.close_window(window)


def test_picking_a_unit_in_flat_iso_hits_the_same_unit_the_iso_geometry_draws() -> None:
    window = _flat_iso_window()
    try:
        window.scenario.unit_manager.units[1].append(Unit(UNIT_X, UNIT_Y, CONST))
        window._render_current(reset_view=False)

        index = unit_pick.build_index(window.scenario)
        ox, oy = iso_geometry.tile_screen_origin(UNIT_X, UNIT_Y, 0, window._iso_proj)
        sx, sy = ox + window._iso_proj.half_w, oy + window._iso_proj.half_h

        resolved_tile = iso_geometry.screen_to_tile(sx, sy, window._iso_elevations, window._iso_proj)
        assert resolved_tile == (UNIT_X, UNIT_Y), "fixture assumption: the probed pixel must land on the unit's own tile"

        entry = unit_pick.pick_unit(
            index, "stepped", sx, sy, window._cache.tile_px,
            window.scenario.map_manager.map_width, window.scenario.map_manager.map_height,
            elevations=window._iso_elevations, proj=window._iso_proj,
        )
        assert entry is not None, "the unit must resolve through the stepped pick branch"
        assert (int(entry.unit.x), int(entry.unit.y)) == (UNIT_X, UNIT_Y)
    finally:
        conftest.close_window(window)


def _stepped_scenario():
    """Blank template with a raised 5x5 block, elevation 4, surrounded by
    flat elevation 0 -- a genuine STEP, unlike _raise_all_tiles' uniform
    elevation above. A uniform field has no elevation DIFFERENCE anywhere,
    so it cannot reproduce the skirt/seam/shadow regression this test
    guards against even with the bug present."""
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

    scenario = load_map_and_units(BLANK_TEMPLATE_PATH)
    mm = scenario.map_manager
    for x in range(58, 63):
        for y in range(58, 63):
            mm.get_tile(x, y).elevation = 4
    return scenario


def test_flat_iso_terrain_has_no_residual_skirt_at_a_real_elevation_step() -> None:
    """The in-game regression this pins: sprites rendered flat, but terrain
    still drew a Stepped-looking grid of skirts/shadows/seams.

    Root cause was `_render_tile_iso` (and its farm-outline twin) computing
    every delta as `tile.elevation - elevations[neighbour]` -- the scenario's
    REAL elevation for the tile itself, mixed against the forced-zero
    NEIGHBOUR read from the (correctly) zeroed array. Both numerator and
    denominator must come from the same (zeroed) array for the delta to
    genuinely read 0. This is a pixel-level property the array-is-all-zero
    assertions elsewhere in this module cannot see: the elevations array was
    already all zero in every case above, precisely because that bug never
    touched the array itself, only what the compositor did with it."""
    import descape.render as render

    stepped = _stepped_scenario()
    flat_control = _stepped_scenario()
    for tile in flat_control.map_manager.terrain:
        tile.elevation = 0

    elevations, proj = render.elevations_and_proj(stepped)
    assert {int(e) for e in elevations.flatten()} == {0, 4}, (
        "fixture assumption: a genuine elevation step must exist to reproduce this bug at all"
    )
    tile_px = render.tile_pixels_for_map(stepped.map_manager.map_width, stepped.map_manager.map_height)
    canvas_w, canvas_h = render._canvas_pixel_dims(proj)

    # A tight rect around the raised block only, generous enough to catch a
    # skirt/shadow reaching outward from it -- not the whole canvas.
    ox, oy = iso_geometry.tile_screen_origin(55, 55, 0, proj)
    x0, y0 = max(0, ox - 400), max(0, oy - 600)
    x1, y1 = min(canvas_w, ox + 1200), min(canvas_h, oy + 600)

    zeroed = np.zeros_like(elevations)
    rendered_flat_iso = render.composite_rect_iso(
        stepped, x0, y0, x1, y1, zeroed, proj, tile_px, {}, {}, with_units=False,
    )
    control_elevations, control_proj = render.elevations_and_proj(flat_control)
    assert control_proj == proj, "fixture assumption: same map size must yield the same projection"
    rendered_control = render.composite_rect_iso(
        flat_control, x0, y0, x1, y1, control_elevations, proj, tile_px, {}, {}, with_units=False,
    )

    assert np.array_equal(rendered_flat_iso, rendered_control), (
        "Flat + Isometric View drew a skirt/seam/shadow that a genuinely flat scenario would "
        "never draw -- the exact corruption the delta-mixing bug produced"
    )
