"""Elevate and Set Elevation skip their live repaint in Flat (GH #78's "Elevate
lags even in Flat").

Flat draws no elevation cue in either of its render paths: top-down
(FlatChunkCache) has no elevation term at all, and Flat + Isometric View
(an IsoChunkCache on an all-zero array) never reads tile.elevation. So a
stroke that writes only elevation recomposites pixels that come out
byte-identical, and ViewerWindow._apply_stroke_dirty skips _apply_dirty for
those tools there. The first test pins the invariant that skip rests on, so a
future Flat elevation cue fails here instead of leaving stale pixels behind.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sprite_edit_bbox import CONST, Unit, sprite_install  # noqa: F401 -- fixture

from descape import iso_geometry, render
from descape.brush import BRUSH_SHAPE_SQUARE, brush_tiles
from descape.elevation_tools import set_tiles_elevation
from descape.render_cache import FlatChunkCache, IsoChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

RAISE_AT = (60, 60)
LOWER_AT = (65, 57)
BRUSH = 3
# Tiles around the edits the rendered windows cover: the brush, its propagation, and margin.
WINDOW_RADIUS = 10


def _spread_scenario():
    """The blank template on a real multi-level elevation spread, with a
    sprite-bearing unit on each footprint about to be edited."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 1 + (tile.x // 3 + tile.y // 5) % 4
    for x, y in (RAISE_AT, LOWER_AT):
        scenario.unit_manager.units[1].append(Unit(x + 0.5, y + 0.5, CONST))
    return scenario


def _edit(scenario) -> None:
    """Raise one brush footprint and lower another, the way Elevate does."""
    mm = scenario.map_manager
    for (cx, cy), delta in ((RAISE_AT, 1), (LOWER_AT, -1)):
        footprint = brush_tiles(cx, cy, BRUSH, BRUSH_SHAPE_SQUARE, mm.map_width, mm.map_height)
        set_tiles_elevation(mm, [(x, y, mm.get_tile(x, y).elevation + delta) for x, y in footprint])


def _iso_window(proj, canvas_dims):
    ex, ey = RAISE_AT
    pad_l, pad_u, pad_r, pad_d = render._sprite_reach_px(proj)
    bounds = [
        iso_geometry.tile_screen_bounds_swept(x, y, proj)
        for x in range(ex - WINDOW_RADIUS, ex + WINDOW_RADIUS + 1)
        for y in range(ey - WINDOW_RADIUS, ey + WINDOW_RADIUS + 1)
    ]
    cw, ch = canvas_dims
    return (
        max(0, min(b[0] for b in bounds) - pad_l),
        max(0, min(b[1] for b in bounds) - pad_u),
        min(cw, max(b[2] for b in bounds) + pad_r),
        min(ch, max(b[3] for b in bounds) + pad_d),
    )


def _render(style: str, scenario) -> np.ndarray:
    """A fresh cache for `style`, built as ViewerWindow._render_current builds
    it, rendered over a fixed window around the edits."""
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "flat":
        cache = FlatChunkCache(scenario, tile_px, sprites=True)
        ex, ey = RAISE_AT
        r = WINDOW_RADIUS
        return cache.render_rect((ex - r) * tile_px, (ey - r) * tile_px, (ex + r + 1) * tile_px, (ey + r + 1) * tile_px).copy()
    elevations, proj = render.elevations_and_proj(scenario)
    if style == "flat-iso":
        elevations = np.zeros_like(elevations)
    cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)
    assert cache._level(0).sprites.skip_ids, "fixture assumption: the units must resolve real sprites"
    return cache.render_rect(*_iso_window(proj, cache.canvas_dims(0))).copy()


@pytest.mark.usefixtures("sprite_install")
def test_an_elevation_edit_changes_no_flat_pixel() -> None:
    """The invariant the skip rests on, for both Flat render paths, with the
    same edit in Stepped as the non-vacuity control."""
    scenario = _spread_scenario()
    mm = scenario.map_manager
    assert len({t.elevation for t in mm.terrain}) == 4, "fixture assumption: a real elevation spread"
    before_elev = [t.elevation for t in mm.terrain]
    before = {style: _render(style, scenario) for style in ("stepped", "flat", "flat-iso")}

    _edit(scenario)
    changed = sum(a != t.elevation for a, t in zip(before_elev, mm.terrain, strict=True))
    assert changed >= 2 * BRUSH * BRUSH, "fixture assumption: both footprints must really change elevation"

    after = {style: _render(style, scenario) for style in ("stepped", "flat", "flat-iso")}
    assert not np.array_equal(before["stepped"], after["stepped"]), (
        "non-vacuity: the same edit must change Stepped's pixels in this window"
    )
    assert np.array_equal(before["flat"], after["flat"]), "top-down Flat drew an elevation change"
    assert np.array_equal(before["flat-iso"], after["flat-iso"]), "Flat + Isometric View drew an elevation change"


def _window(style: str, tool: str):
    """A Terrain-mode blank window on `tool`, in `style` ("flat-iso", "flat" or
    "stepped"). Isometric View is set before the style switch, as
    tests/test_brush_stroke.py's _shown_flat() explains."""
    window = conftest.terrain_edit_window()
    window._on_tool_selected(tool)
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window.brush_size_spin.setValue(BRUSH)
    window.elevation_level_spin.setValue(4)
    if style != "stepped":
        window.iso_action.setChecked(style == "flat-iso")
        window.terrain_style_combo.setCurrentText("Flat")
    assert window._current_tool == tool, f"fixture assumption: {tool} must stay selected in {style}"
    expected_cache = FlatChunkCache if style == "flat" else IsoChunkCache
    assert isinstance(window._cache, expected_cache), f"fixture assumption: {style} builds {expected_cache.__name__}"
    return window


def _spy(window, name: str) -> list:
    calls: list = []
    original = getattr(window, name)

    def recording(*args) -> None:
        calls.append(args)
        original(*args)

    setattr(window, name, recording)
    return calls


def _drag(window) -> None:
    window.on_edit_stroke_start()
    for k in range(4):
        window.on_edit_stroke_tiles([(40 + 2 * k, 40), (41 + 2 * k, 40)], 0)
    window.on_edit_stroke_end()


@pytest.mark.parametrize("tool", ["elevation", "set_level"])
@pytest.mark.parametrize("style", ["flat-iso", "flat", "stepped"])
def test_an_elevation_stroke_skips_the_repaint_only_in_flat(style, tool) -> None:
    window = _window(style, tool)
    try:
        mm = window.scenario.map_manager
        start = [t.elevation for t in mm.terrain]
        apply_calls = _spy(window, "_apply_dirty")
        cancel_calls = _spy(window, "_cancel_warms")
        rearm_calls = _spy(window, "_start_level_warm")
        try:
            _drag(window)
        finally:
            del window._apply_dirty, window._cancel_warms, window._start_level_warm

        after = [t.elevation for t in mm.terrain]
        assert after != start, "the stroke must really change elevation"
        if style == "stepped":
            assert apply_calls, "control: Stepped draws elevation, so its stroke must repaint"
            return
        assert apply_calls == [], f"the {tool} stroke in {style} repainted, but no Flat pixel can change"
        assert cancel_calls and len(rearm_calls) == len(cancel_calls), "the skip must still cancel and re-arm the warms"
        if style == "flat-iso":
            assert np.all(window._iso_elevations == 0), "Flat + Isometric View's array must stay all zero"
        else:
            assert window._iso_elevations is None

        assert len(window.edit_history.records) == 1, "the stroke must still commit one undo record"
        window.undo()
        assert [t.elevation for t in mm.terrain] == start, "undo must restore the stroke-start elevations"
        if style == "flat-iso":
            assert np.all(window._iso_elevations == 0)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["flat-iso", "flat"])
def test_switching_to_stepped_after_a_flat_elevate_shows_the_real_heights(style) -> None:
    """The skipped repaint leaves nothing stale: Stepped rebuilds from the
    scenario, and an undo there clears the raise again."""
    window = _window(style, "elevation")
    try:
        mm = window.scenario.map_manager
        _drag(window)
        window.terrain_style_combo.setCurrentText("Stepped")
        real, _ = render.elevations_and_proj(window.scenario)
        assert real.any(), "fixture assumption: the Flat stroke must have raised something"
        assert np.array_equal(window._iso_elevations, real), "Stepped must show the heights the Flat stroke wrote"

        window.undo()
        assert all(t.elevation == 0 for t in mm.terrain)
        assert not window._iso_elevations.any(), "undo in Stepped must clear the raise from the render array"
    finally:
        conftest.close_window(window)
