"""descape/terrain_units.py's ViewerWindow wiring: the Trees/Eye candy
checkboxes, the composite-undo path Draw and Paint Can both feed into, the
Paint Can large-fill confirm guard, and settings persistence. Same offscreen
technique tests/test_fill_tool.py documents; every ViewerWindow() here must
call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from test_sprite_toggle_viewer import sprite_install  # noqa: F401 -- fixture, requested by name

# Fixtures and a constant: blend_install is requested by name and requests the next two itself;
# blend_keys_match_the_mirrors is autouse, so every composite here checks the cache's blend key map.
from test_terrain_blend_render import (  # noqa: F401
    LAYER_ID,
    blend_install,
    blend_keys_match_the_mirrors,
    fake_farm,
    synthetic_textures,
)

from descape import render_cache, settings
from descape.edit_history import CompositeDiffRecord, TileDiffRecord

import conftest
from test_unit_sprites import CONST

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FOREST_OAK = 10  # density 1000/1000 -- deterministic single tree per tile
FOREST_OAK_CONST = 411


def _window(tool: str = "draw"):
    """The blank template loaded, Terrain mode, `tool` active, FOREST_OAK
    selected in the terrain panel. Caller must edit_history.mark_saved() +
    close()."""
    window = conftest.terrain_edit_window()
    window._on_tool_selected(tool)
    window.terrain_panel.set_terrain(FOREST_OAK)
    return window


def _gaia_units_at(window, x: int, y: int):
    return [u for u in window.scenario.unit_manager.units[0] if int(u.x) == x and int(u.y) == y]


def test_draw_with_trees_on_adds_a_unit_as_one_composite_undo_step() -> None:
    window = _window("draw")
    try:
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(False)

        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(5, 5, 0)
        window.on_edit_stroke_end()

        assert len(window.edit_history.records) == 1
        record = window.edit_history.records[-1]
        assert isinstance(record, CompositeDiffRecord)
        assert record.label == "Paint terrain"

        units = _gaia_units_at(window, 5, 5)
        assert len(units) == 1
        assert units[0].unit_const == FOREST_OAK_CONST
        mm = window.scenario.map_manager
        assert mm.get_tile(5, 5).terrain_id == FOREST_OAK

        # One Ctrl+Z reverts both terrain and the tree.
        window.undo()
        assert mm.get_tile(5, 5).terrain_id != FOREST_OAK
        assert _gaia_units_at(window, 5, 5) == []

        # Ctrl+Y restores both.
        window.redo()
        assert mm.get_tile(5, 5).terrain_id == FOREST_OAK
        assert _gaia_units_at(window, 5, 5)[0].unit_const == FOREST_OAK_CONST
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_auto_placed_trees_survive_a_save_and_reload_with_varied_rotation(tmp_path, monkeypatch) -> None:
    """GH #89 step 5's DEscape half. The variant RNG is unseeded, so 40 tiles
    make "more than one distinct rotation" near-certain without pinning values."""
    import descape.viewer as viewer_module
    from descape.scenario_io import load_map_and_units

    dest = tmp_path / "trees.aoe2scenario"
    monkeypatch.setattr(
        viewer_module.QFileDialog, "getSaveFileName", staticmethod(lambda *args, **kwargs: (str(dest), ""))
    )
    window = _window("draw")
    try:
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(False)
        window.on_edit_stroke_start()
        for y in range(10, 14):
            for x in range(10, 20):
                window.on_edit_stroke_tile(x, y, 0)
        window.on_edit_stroke_end()

        placed = sorted(
            (int(u.x), int(u.y), u.rotation)
            for u in window.scenario.unit_manager.units[0]
            if u.unit_const == FOREST_OAK_CONST
        )
        assert len(placed) == 40
        window.save_as()
        assert dest.is_file()
    finally:
        window.edit_history.mark_saved()
        window.close()

    reloaded = load_map_and_units(dest)
    read_back = sorted(
        (int(u.x), int(u.y), u.rotation) for u in reloaded.unit_manager.units[0] if u.unit_const == FOREST_OAK_CONST
    )
    assert read_back == placed
    assert len({rotation for _x, _y, rotation in read_back}) > 1, "every tree read back with one rotation"


def test_draw_with_trees_off_pushes_a_plain_tile_record() -> None:
    window = _window("draw")
    try:
        window.paint_trees_check.setChecked(False)
        window.paint_eye_candy_check.setChecked(False)

        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(5, 5, 0)
        window.on_edit_stroke_end()

        assert len(window.edit_history.records) == 1
        assert type(window.edit_history.records[-1]) is TileDiffRecord
        assert _gaia_units_at(window, 5, 5) == []
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_repainting_the_same_forest_terrain_adds_no_new_undo_step() -> None:
    """§3's own guarantee: a tile already carrying the terrain being painted
    plans nothing, so a second pass over the same tile with the same
    terrain must not re-roll a variant or grow the undo stack."""
    window = _window("draw")
    try:
        window.paint_trees_check.setChecked(True)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(5, 5, 0)
        window.on_edit_stroke_end()
        assert len(window.edit_history.records) == 1
        first_variant = _gaia_units_at(window, 5, 5)[0].rotation

        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(5, 5, 0)
        window.on_edit_stroke_end()

        assert len(window.edit_history.records) == 1  # no phantom step
        units = _gaia_units_at(window, 5, 5)
        assert len(units) == 1
        assert units[0].rotation == first_variant  # not re-rolled
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paint_can_below_threshold_fills_units_with_no_confirm(monkeypatch) -> None:
    import descape.viewer as viewer_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("QMessageBox.question must not be called under the threshold")

    monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fail_if_called))
    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 100_000)

    window = _window("fill")
    try:
        window.paint_trees_check.setChecked(True)
        window.on_fill(0, 0, 0)

        record = window.edit_history.records[-1]
        assert isinstance(record, CompositeDiffRecord)
        mm = window.scenario.map_manager
        assert all(t.terrain_id == FOREST_OAK for t in mm.terrain)
        oak_units = [u for u in window.scenario.unit_manager.units[0] if u.unit_const == FOREST_OAK_CONST]
        assert len(oak_units) == mm.map_width * mm.map_height

        window.undo()
        assert not [u for u in window.scenario.unit_manager.units[0] if u.unit_const == FOREST_OAK_CONST]
        assert all(t.terrain_id != FOREST_OAK for t in mm.terrain)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paint_can_above_threshold_confirms_and_cancel_leaves_map_untouched(monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 10)

    window = _window("fill")
    try:
        window.paint_trees_check.setChecked(True)
        calls = []

        def cancel(*args, **kwargs):
            calls.append(args)
            return QMessageBox.Cancel

        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(cancel))
        window.on_fill(0, 0, 0)

        assert len(calls) == 1
        assert window.edit_history.records == []
        mm = window.scenario.map_manager
        assert all(t.terrain_id != FOREST_OAK for t in mm.terrain)

        monkeypatch.setattr(
            viewer_module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
        )
        window.on_fill(0, 0, 0)
        assert len(window.edit_history.records) == 1
        assert all(t.terrain_id == FOREST_OAK for t in mm.terrain)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paint_can_confirm_is_skipped_with_both_checkboxes_off(monkeypatch) -> None:
    """A terrain-only fill has no per-tile unit cost, so the guard must not
    fire even on a huge region -- pinned by making QMessageBox.question
    raise if it's ever reached."""
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 1)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("QMessageBox.question must not be called with both checkboxes off")

    monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fail_if_called))

    window = _window("fill")
    try:
        window.paint_trees_check.setChecked(False)
        window.paint_eye_candy_check.setChecked(False)
        window.on_fill(0, 0, 0)
        mm = window.scenario.map_manager
        assert all(t.terrain_id == FOREST_OAK for t in mm.terrain)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_checkbox_visibility_follows_the_terrain_param() -> None:
    window = _window("draw")
    try:
        assert window.paint_trees_param_action.isVisible()
        assert window.paint_eye_candy_param_action.isVisible()

        window._on_tool_selected("elevation")  # no terrain param at all
        assert not window.paint_trees_param_action.isVisible()
        assert not window.paint_eye_candy_param_action.isVisible()

        window._on_tool_selected("fill")
        assert window.paint_trees_param_action.isVisible()
        assert window.paint_eye_candy_param_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paint_trees_checkbox_defaults_and_persists() -> None:
    assert settings.get_paint_trees() is True
    assert settings.get_paint_eye_candy() is False

    window = _window("draw")
    try:
        assert window.paint_trees_check.isChecked() is True
        assert window.paint_eye_candy_check.isChecked() is False

        window.paint_trees_check.setChecked(False)
        window.paint_eye_candy_check.setChecked(True)
        assert settings.get_paint_trees() is False
        assert settings.get_paint_eye_candy() is True
    finally:
        window.edit_history.mark_saved()
        window.close()

    # A fresh window picks up the persisted values.
    window2 = _window("draw")
    try:
        assert window2.paint_trees_check.isChecked() is False
        assert window2.paint_eye_candy_check.isChecked() is True
    finally:
        window2.edit_history.mark_saved()
        window2.close()


# --- Stroke-end splice (stroke-end repaint splice plan, Steps 3, 5 and 6) ---

FOREST_PALM = 13  # density 1000/1000, a different tree, so B replaces every A tree it crosses
STROKE_TILES = [(20 + i, 30) for i in range(8)]
# Half over STROKE_TILES (remove + add per tile), half onto grass, so the swap shows even without sprite art.
SHIFTED_TILES = [(24 + i, 30) for i in range(8)]


def _styled_window(style: str):
    """`style`, or "FlatIso": Flat with Isometric View left on, which renders
    through Stepped's cache over all-zero elevations."""
    window = _window("draw")
    if style == "Flat":
        # Top-down Flat, not the Flat+Isometric path (see test_unit_edit_viewer's _window).
        window.iso_action.setChecked(False)
    if style == "FlatIso":
        assert window.iso_action.isChecked()
        style = "Flat"
    window.terrain_style_combo.setCurrentText(style)
    window.paint_trees_check.setChecked(True)
    window.paint_eye_candy_check.setChecked(False)
    return window


def _draw(window, terrain_id: int, tiles=STROKE_TILES) -> None:
    window.terrain_panel.set_terrain(terrain_id)
    window.on_edit_stroke_start()
    for x, y in tiles:
        window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()


def _canvas(window):
    canvas_w, canvas_h = window._cache.canvas_dims(0)
    return window._cache.render_rect(0, 0, canvas_w, canvas_h, mip=0).copy()


def _fresh(window, style: str):
    from descape import render
    from descape.render_cache import FlatChunkCache

    canvas_w, canvas_h = window._cache.canvas_dims(0)
    sprites = window._cache.sprites_enabled
    if style == "Flat":
        mm = window.scenario.map_manager
        fresh = FlatChunkCache(
            window.scenario, render.tile_pixels_for_map(mm.map_width, mm.map_height),
            sprites=sprites, unit_filter=window._cache.unit_filter,
        )
        return fresh.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    if style == "Stepped":
        full = render.render_terrain_iso_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    else:
        full = render.render_terrain_sloped_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    return full[:canvas_h, :canvas_w]


def _assert_numpy_fresh(window, style: str, pixels, step: str) -> None:
    """With native active, `pixels` equal _fresh() on the numpy backend, so
    native is not only checked against native. A no-op on numpy."""
    from descape import composite_backend

    if os.environ.get("DESCAPE_REQUIRE_NATIVE") == "1":
        assert composite_backend.active_backend() == "native", composite_backend.unavailable_reason
    if composite_backend.active_backend() != "native":
        return
    with composite_backend.use_backend("numpy"):
        reference = _fresh(window, style)
    assert np.array_equal(pixels, reference), f"{step}: mip 0 differs from a fresh numpy render"


def _spy(window, monkeypatch):
    """Records every _after_unit_mutation `changed` argument, every wholesale
    source rebuild (a _refresh_source_caches() call from inside
    invalidate_units(), which every style's splice path never makes; patch()
    and Flat's patch_rects() call it too, so only calls inside count) and
    every in-place units_by_tile update that succeeded."""
    calls = {"changed": [], "wholesale": 0, "in_place": 0}
    inside = [False]
    real_after = window._after_unit_mutation
    cache = window._cache
    real_invalidate, real_refresh = cache.invalidate_units, cache._refresh_source_caches
    real_in_place = getattr(cache, "_update_units_by_tile_in_place", None)

    def after(changed=None, **kwargs):
        calls["changed"].append(changed)
        return real_after(changed, **kwargs)

    def invalidate(changed=None, splice_levels=True, **kwargs):
        inside[0] = True
        try:
            return real_invalidate(changed, splice_levels=splice_levels, **kwargs)
        finally:
            inside[0] = False

    def refresh(elevation_changed=None, **kwargs):
        calls["wholesale"] += inside[0]
        return real_refresh(elevation_changed, **kwargs)

    def in_place(changed):
        done = real_in_place(changed)
        calls["in_place"] += done
        return done

    monkeypatch.setattr(window, "_after_unit_mutation", after)
    monkeypatch.setattr(cache, "invalidate_units", invalidate)
    monkeypatch.setattr(cache, "_refresh_source_caches", refresh)
    monkeypatch.setattr(cache, "_update_units_by_tile_in_place", in_place)
    return calls


def _splice_any_batch(monkeypatch) -> None:
    """The blank template has too few units for a batch to beat Flat's
    wholesale rebuild (viewer._SPLICE_COST_RATIO), and a whole resident canvas
    against a small offscreen viewport would make Stepped/Sloped evict rather
    than patch (viewer._TIGHT_PATCH_AREA_RATIO), so pin both off."""
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "_SPLICE_COST_RATIO", 0)
    _pin_area_ratio(monkeypatch, float("inf"))


def _pin_area_ratio(monkeypatch, value: float, rects: float = 0.0) -> None:
    """Sets the batch patch-or-evict area ratio, for every style, and
    Stepped's per-chunk rect ratio past it (TASK-031.61 C1). `rects` 0, the
    default, keeps an over-ratio batch evicting; inf always patches rects."""
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "_TIGHT_PATCH_AREA_RATIO", {"stepped": value, "sloped": value})
    monkeypatch.setattr(viewer_module, "_RECT_PATCH_AREA_RATIO", rects)


@pytest.mark.parametrize("sprites", [False, True], ids=["marks", "sprites"])
@pytest.mark.parametrize("size", [1, 9])
@pytest.mark.parametrize("style", ["Stepped", "Sloped", "Flat", "FlatIso"])
def test_draw_through_existing_trees_splices_and_repaints_like_a_fresh_render(
    style, size, sprites, monkeypatch, request
) -> None:
    """Plan Verification 1: brush 1 and 9 through a forest, every style, tree sprites off and on.
    - Stroke B's end splices (a non-empty `changed`) with a remove + add on shared tiles.
    - Brush 9's batch is wider than brush 1's (over twice the stroke's 8 tiles), so the size applied.
    - No wholesale source rebuild at the stroke end or its redo.
    - The undo splices too, except Flat 2D at brush 9 (A's oaks go back mid-list: the tail-insert guard).
    - mip 0 changes and equals a fresh render after the stroke, the undo and the redo.
    - With native active, mip 0 also equals a fresh numpy render after the stroke, and at brush 9
      after its undo and redo too (each numpy render costs ~0.7 s)."""
    if sprites:
        request.getfixturevalue("tree_sprites")
    _splice_any_batch(monkeypatch)
    window = _styled_window(style)
    fresh_style = "Stepped" if style == "FlatIso" else style
    try:
        window.show_sprites_action.setChecked(sprites)
        assert window._cache.sprites_enabled is sprites
        window.brush_size_spin.setValue(size)
        _draw(window, FOREST_OAK)
        before = _canvas(window)  # every chunk resident, so a stale one would show
        assert np.array_equal(before, _fresh(window, fresh_style))
        calls = _spy(window, monkeypatch)

        _draw(window, FOREST_PALM, SHIFTED_TILES)

        (changed,) = calls["changed"]
        assert changed, "the stroke end took the wholesale unit invalidation"
        # Brush 1: 4 removes + 8 adds; brush 9 sweeps a 9-tile-tall band.
        assert (len(changed) > 2 * len(SHIFTED_TILES)) is (size > 1), len(changed)
        removed = {s.old_own_tile for s in changed if s.new_own_tile is None}
        assert removed and removed <= {s.new_own_tile for s in changed}, "expected remove + add on shared tiles"
        assert calls["wholesale"] == 0, "the cache fell back to a wholesale source rebuild"
        after = _canvas(window)
        assert not np.array_equal(after, before), "the new trees changed nothing on screen"
        assert np.array_equal(after, _fresh(window, fresh_style))
        _assert_numpy_fresh(window, fresh_style, after, "stroke")

        window.undo()
        mid_list = style == "Flat" and size > 1
        assert bool(calls["changed"][-1]) is not mid_list, f"undo took {calls['changed'][-1]}"
        undo_wholesale = calls["wholesale"]
        assert undo_wholesale == 0 or mid_list
        assert np.array_equal(_canvas(window), _fresh(window, fresh_style))
        assert np.array_equal(_canvas(window), before)
        if size > 1:
            _assert_numpy_fresh(window, fresh_style, _canvas(window), "undo")
        window.redo()
        assert calls["changed"][-1], "redo took the wholesale unit invalidation"
        assert calls["wholesale"] == undo_wholesale
        assert np.array_equal(_canvas(window), after)
        if size > 1:
            _assert_numpy_fresh(window, fresh_style, _canvas(window), "redo")
    finally:
        window.edit_history.mark_saved()
        window.close()


def _spy_evictions(window, monkeypatch) -> list:
    """Every bbox window._cache.invalidate_region() evicts from here on at the
    visible level (levels None included). A unit edit's eviction of today's
    bbox from the other resident levels (_repaint_unit_edit_split()) is not
    the patch-or-evict choice these tests pin, so it is left out."""
    evictions = []
    real_evict = window._cache.invalidate_region

    def evict(bbox, levels=None, **kwargs):
        target = window.map_view.viewport_chunk_target()
        if levels is None or target is None or target[0] in levels:
            evictions.append(bbox)
        return real_evict(bbox, levels=levels, **kwargs)

    monkeypatch.setattr(window._cache, "invalidate_region", evict)
    return evictions


@pytest.mark.parametrize("path", ["component", "wholesale"])
@pytest.mark.parametrize("repaint", ["patch", "evict"])
@pytest.mark.parametrize("sprites", [False, True], ids=["marks", "sprites"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_draw_over_a_tile_another_unit_holds_repaints_only_its_bbox(
    style, sprites, repaint, path, monkeypatch, request
) -> None:
    """A contested tile makes the batch guard refuse the splice. The cache
    splices the shared tile's component, or with the component cap at 0
    (`wholesale`) updates units_by_tile in place, the batch being pure adds or
    removals, and rebuilds the levels (from the sprite memo); either way the stroke end
    repaints only the batch's bbox, eagerly or by evicting its chunks: never
    the whole canvas. Undo and redo take the same path, the component one
    through _gaia_membership_diff(). With sprites, the contested unit is a
    real sprite, so the re-derived draw is what the pixel oracle checks."""
    if sprites:
        request.getfixturevalue("sprite_install")
    if path == "wholesale":
        monkeypatch.setattr(render_cache, "_COMPONENT_SPLICE_MAX_UNITS", 0)
        monkeypatch.setattr(render_cache, "VISIBLE_SPLICE_MAX_UNITS", 0)
    _splice_any_batch(monkeypatch)
    if repaint == "evict":
        _pin_area_ratio(monkeypatch, 0)
    window = _styled_window(style)
    try:
        # A configured real install turns sprites on by default, so set both states explicitly.
        window.show_sprites_action.setChecked(sprites)
        assert window._cache.sprites_enabled is sprites
        # Show mip 0, the level _canvas() keeps resident: the repaint choice pinned below is the visible level's.
        _cx0, _cy0, cx1, cy1 = window._cache.chunk_index_range(0, 0, 0, *window._cache.canvas_dims(0))
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (0, 0, 0, cx1, cy1))
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [1]):
            model.add(1, CONST, STROKE_TILES[3][0] + 0.5, STROKE_TILES[3][1] + 0.5)
        before = _canvas(window)
        calls = _spy(window, monkeypatch)
        canvas = (0, 0, *window._cache.canvas_dims(0))
        evictions = _spy_evictions(window, monkeypatch)

        _draw(window, FOREST_OAK)

        (changed,) = calls["changed"]
        assert changed, "the stroke end took the whole-canvas path"
        # With the component cap at 0 the refused batch is pure GAIA adds (its
        # undo pure removals), so it updates units_by_tile in place, not wholesale.
        in_place = 1 if path == "wholesale" else 0
        assert calls["wholesale"] == 0, "a membership-only batch took the wholesale source rebuild"
        assert calls["in_place"] == in_place, f"expected the contested batch to take the {path} path"
        assert bool(evictions) is (repaint == "evict")
        assert canvas not in evictions
        after = _canvas(window)
        assert not np.array_equal(after, before)
        assert np.array_equal(after, _fresh(window, style))
        window.undo()
        assert calls["changed"][-1]
        assert calls["wholesale"] == 0
        assert calls["in_place"] == 2 * in_place, f"the undo did not take the {path} path"
        assert np.array_equal(_canvas(window), before)
        assert np.array_equal(_canvas(window), _fresh(window, style))
        window.redo()
        assert calls["changed"][-1]
        assert calls["wholesale"] == 0
        assert calls["in_place"] == 3 * in_place, f"the redo did not take the {path} path"
        assert canvas not in evictions
        assert np.array_equal(_canvas(window), after)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_stepped_stroke_whose_patch_outgrows_the_viewport_evicts_only_its_bbox(monkeypatch) -> None:
    """At the default area ratio, a long stroke over a fully resident canvas
    seen through a one-chunk viewport evicts the chunks its bbox touches and
    keeps the rest, rather than patching eagerly or evicting everything."""
    window = _styled_window("Stepped")
    try:
        _canvas(window)
        resident = len(window._cache._cache)
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (0, 0, 0, 0, 0))
        calls = _spy(window, monkeypatch)
        evictions = _spy_evictions(window, monkeypatch)
        _draw(window, FOREST_OAK, [(10 + i, 10 + i) for i in range(60)])
        (changed,) = calls["changed"]
        assert changed
        (bbox,) = evictions
        assert bbox != (0, 0, *window._cache.canvas_dims(0))
        assert 0 < len(window._cache._cache) < resident
        assert np.array_equal(_canvas(window), _fresh(window, "Stepped"))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_sloped_stroke_patches_its_bbox_eagerly(monkeypatch) -> None:
    """Sloped's measured crossover is past every stroke benched, so at the
    default ratio STROKE_TILES' bbox is patched in the handler, not evicted."""
    window = _styled_window("Sloped")
    try:
        _canvas(window)
        calls = _spy(window, monkeypatch)
        evictions = _spy_evictions(window, monkeypatch)
        _draw(window, FOREST_OAK)
        (changed,) = calls["changed"]
        assert changed
        assert evictions == []
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_single_unit_edit_always_patches(monkeypatch) -> None:
    """The area rule is for batch callers only: a single-unit tool keeps its
    eager patch even when every batch would evict."""
    _pin_area_ratio(monkeypatch, 0)
    window = _styled_window("Stepped")
    try:
        _canvas(window)
        calls = _spy(window, monkeypatch)
        evictions = _spy_evictions(window, monkeypatch)
        from descape.render_cache import UnitSplice

        model = window._ensure_unit_edits()
        splices: list = []
        with window._unit_edit(model, "Add", [1], splices=splices):
            unit = model.add(1, CONST, 40.5, 40.5)
            own, tiles = window._unit_footprint(unit)
            splices.append(UnitSplice(1, len(window.scenario.unit_manager.units[1]) - 1, unit, None, own, (), tiles))
        (changed,) = calls["changed"]
        assert changed
        assert evictions == []
        assert np.array_equal(_canvas(window), _fresh(window, "Stepped"))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_batch_too_large_for_the_map_takes_the_wholesale_path_in_flat(monkeypatch) -> None:
    """Flat has no memo: at the default ratio, a stroke's trees on a
    near-empty map cost more to patch eagerly than the wholesale rebuild
    they would save."""
    window = _styled_window("Flat")
    try:
        calls = _spy(window, monkeypatch)
        _draw(window, FOREST_OAK)
        assert calls["changed"] == [None]
    finally:
        window.edit_history.mark_saved()
        window.close()


def _prime_ref_index(window):
    """The trigger reference index built and held, so an edit must patch or drop it."""
    index = window._unit_reference_index()
    assert index is window._unit_ref_index is not None
    return index


def _assert_ref_index_current(window, step: str) -> None:
    """Whatever the edit did to the held index (patched it, or dropped it for a
    lazy rebuild), the next read equals a fresh build."""
    from descape import unit_references

    index = window._unit_reference_index()
    fresh = unit_references.build_reference_index(window.scenario)
    assert dict(index.by_id) == dict(fresh.by_id), f"{step}: the reference index is stale"
    assert index.duplicates == fresh.duplicates, step


@pytest.mark.parametrize("style", ["Stepped", "Sloped", "Flat"])
def test_undo_and_redo_of_a_non_gaia_place_take_the_paste_shaped_path_except_on_flat(style, monkeypatch) -> None:
    """A player-1 Place is a pure tail append, so its undo and redo take the
    scoped membership path (_membership_diff()'s paste shape) on Stepped and
    Sloped, and stay wholesale on Flat. Pixels and the reference index match
    a fresh build after each."""
    window = _styled_window(style)
    try:
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [1]):
            model.add(1, 83, 40.5, 40.5)
        _prime_ref_index(window)
        calls = _spy(window, monkeypatch)
        for step, act in (("undo", window.undo), ("redo", window.redo)):
            _canvas(window)
            act()
            changed = calls["changed"][-1]
            if style == "Flat":
                assert changed is None, f"{step}: Flat took the scoped path"
            else:
                assert changed and len(changed) == 1 and changed[0].player_id == 1, f"{step}: took {changed}"
            assert np.array_equal(_canvas(window), _fresh(window, style)), step
            _assert_ref_index_current(window, step)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("last", [True, False], ids=["last-unit", "mid-list"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_undo_and_redo_of_a_convert_stay_wholesale(style, last, monkeypatch) -> None:
    """A Convert moves a unit between lists: a mid-list one is no tail change,
    and the list's last one is on both sides of the diff, so neither is
    paste-shaped. Pixels and the reference index match a fresh build."""
    window = _styled_window(style)
    try:
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [1]):
            for i in range(3):
                model.add(1, 83, 40.5 + 2 * i, 40.5)
        target = window.scenario.unit_manager.units[1][-1 if last else 1]
        with window._unit_edit(model, "Convert", [1, 2]):
            model.reassign(target, 2)
        _prime_ref_index(window)
        calls = _spy(window, monkeypatch)
        for step, act in (("undo", window.undo), ("redo", window.redo)):
            _canvas(window)
            act()
            assert calls["changed"][-1] is None, f"{step}: a Convert left the wholesale path"
            assert np.array_equal(_canvas(window), _fresh(window, style)), step
            _assert_ref_index_current(window, step)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_draw_undo_with_no_elevation_keeps_the_tiles_first_scoped_patch(monkeypatch) -> None:
    """Draw's terrain + GAIA composite carries no elevation change, so its undo
    and redo keep the tiles-first order and _patch_unit_edit_cache(), not the
    sources-first branch Paste and Mirror Map take."""
    _splice_any_batch(monkeypatch)
    window = _styled_window("Stepped")
    try:
        _draw(window, FOREST_OAK)
        scoped = []
        real = window._patch_unit_edit_cache
        monkeypatch.setattr(window, "_patch_unit_edit_cache", lambda *a, **k: (scoped.append(1), real(*a, **k)))
        for step, act in (("undo", window.undo), ("redo", window.redo)):
            del scoped[:]
            act()
            assert scoped == [1], f"{step}: skipped the scoped patch"
            assert np.array_equal(_canvas(window), _fresh(window, "Stepped")), step
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_undo_and_redo_of_a_mirror_map_with_units_and_elevation_run_the_unit_sources_first(style, monkeypatch) -> None:
    """Mirror Map's tile + unit composite changes elevation under its unit
    images, so its undo and redo run the unit sources before the tile patch
    (the order _flush_pending requires), skipping _patch_unit_edit_cache().
    Pixels and the reference index match a fresh build after each."""
    from descape.mirror_tools import plan_mirror, plan_mirror_units

    window = _styled_window(style)
    try:
        window._on_tool_selected("set_level")
        window.elevation_level_spin.setValue(1)
        for y in range(10, 13):
            for x in range(10, 13):
                window.on_edit_stroke_start()
                window.on_edit_stroke_tile(x, y, 0)
                window.on_edit_stroke_end()
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [1]):
            model.add(1, 83, 11.5, 11.5)
        mm = window.scenario.map_manager
        plan = plan_mirror(mm, 1, 0, do_terrain=True, do_elevation=True)
        assert not plan.elevation_violations
        unit_plan = plan_mirror_units(
            mm, 1, 0, window.scenario.unit_manager.units, plan.source_indices,
            referencing=window.unit_edits.referencing,
        )
        assert window.on_mirror(plan, unit_plan)
        _prime_ref_index(window)
        events = []
        real_dirty, real_units = window._apply_dirty, window._cache.invalidate_units
        real_scoped = window._patch_unit_edit_cache
        monkeypatch.setattr(window, "_apply_dirty", lambda d: (events.append("tiles"), real_dirty(d))[1])
        monkeypatch.setattr(
            window._cache, "invalidate_units", lambda *a, **k: (events.append("units"), real_units(*a, **k))[1]
        )
        monkeypatch.setattr(
            window, "_patch_unit_edit_cache", lambda *a, **k: (events.append("scoped"), real_scoped(*a, **k))[1]
        )
        for step, act in (("undo", window.undo), ("redo", window.redo)):
            _canvas(window)
            del events[:]
            act()
            assert events[:2] == ["units", "tiles"] and "scoped" not in events, f"{step}: {events}"
            assert np.array_equal(_canvas(window), _fresh(window, style)), step
            _assert_ref_index_current(window, step)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paint_can_with_trees_still_takes_the_wholesale_path(monkeypatch) -> None:
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 100_000)
    window = _styled_window("Stepped")
    try:
        window._on_tool_selected("fill")
        window.terrain_panel.set_terrain(FOREST_OAK)
        calls = _spy(window, monkeypatch)
        window.on_fill(0, 0, 0)
        assert calls["changed"] == [None]
        assert calls["wholesale"] >= 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_draw_stroke_record_is_covered_by_its_live_repaints(monkeypatch) -> None:
    """What lets the stroke end skip the whole-stroke _apply_dirty(): every
    tile the record changes was already handed to _apply_dirty() by a live
    step, auto-beach ring included."""
    window = _styled_window("Stepped")
    try:
        window.terrain_panel.set_terrain(22)  # WATER_DEEP, so the ring has something to ring
        window.auto_beach_check.setChecked(True)
        window.beach_width_spin.setValue(2)
        live: set[int] = set()
        real_apply = window._apply_dirty

        def apply_dirty(indices):
            live.update(indices)
            return real_apply(indices)

        monkeypatch.setattr(window, "_apply_dirty", apply_dirty)
        captured = []
        real_build = window.edit_history.build_stroke_record

        def build(*args, **kwargs):
            record = real_build(*args, **kwargs)
            captured.append(record)
            live_at_end.update(live)
            return record

        live_at_end: set[int] = set()
        monkeypatch.setattr(window.edit_history, "build_stroke_record", build)
        window.on_edit_stroke_start()
        for x, y in STROKE_TILES:
            window.on_edit_stroke_tile(x, y, 0)
        window.on_edit_stroke_end()

        (record,) = captured
        changed = {i for i, _old, _new in record.changes}
        assert any(new[0] != 22 for _i, _old, new in record.changes), "auto-beach laid no ring"
        assert changed <= live_at_end
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- GH #183: auto-placed trees skip cliff tiles ---

CLIFF_3X2 = 269  # span (3, 2): anchor (11.5, 11.0) on tile (10, 10), so footprint and anchor tile differ
CLIFF_3X3 = 264


def _place_cliff(window, const: int, player: int, tile=(10, 10)):
    from descape import render, terrain_palette

    span = terrain_palette.tile_span(const, (1, 1))
    x, y = render.span_anchor(tile[0], tile[1], *span)
    model = window._ensure_unit_edits()
    assert model is not None
    with window._unit_edit(model, "Place cliff", [player]):
        cliff = model.add(player, const, x, y)
    mm = window.scenario.map_manager
    footprint = set(render.unit_occupied_tiles(cliff, mm.map_width, mm.map_height))
    assert len(footprint) == span[0] * span[1]
    return footprint


def _oak_tiles(window) -> list[tuple[int, int]]:
    return [(int(u.x), int(u.y)) for u in window.scenario.unit_manager.units[0] if u.unit_const == FOREST_OAK_CONST]


@pytest.mark.parametrize(("const", "player"), [(CLIFF_3X2, 0), (CLIFF_3X3, 1)])
def test_drawing_forest_over_a_cliff_plants_no_tree_on_its_footprint(const: int, player: int) -> None:
    window = _window("draw")
    try:
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(False)
        footprint = _place_cliff(window, const, player)
        xs = [x for x, _y in footprint]
        ys = [y for _x, y in footprint]
        beyond = (max(xs) + 1, min(ys))
        window.on_edit_stroke_start()
        for tile in [*sorted(footprint), beyond]:
            window.on_edit_stroke_tile(*tile, 0)
        window.on_edit_stroke_end()

        mm = window.scenario.map_manager
        assert all(mm.get_tile(x, y).terrain_id == FOREST_OAK for x, y in footprint)  # terrain still painted
        oaks = _oak_tiles(window)
        assert not footprint & set(oaks)
        assert oaks == [beyond]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paint_can_forest_plants_no_tree_on_a_cliff_footprint(monkeypatch) -> None:
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 100_000)
    window = _window("fill")
    try:
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(False)
        footprint = _place_cliff(window, CLIFF_3X2, 0)
        window.on_fill(0, 0, 0)

        mm = window.scenario.map_manager
        assert all(t.terrain_id == FOREST_OAK for t in mm.terrain)
        oaks = set(_oak_tiles(window))
        assert not footprint & oaks
        assert len(oaks) == mm.map_width * mm.map_height - len(footprint)
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- Over-cap stroke end (TASK-031.61 A2) ---

OVER_CAP = 4  # UNIT_SPLICE_MAX_UNITS pinned under STROKE_TILES' 8 trees


@pytest.fixture
def tree_sprites(sprite_install, monkeypatch):  # noqa: F811 -- the imported fixture, requested by name
    """sprite_install's synthetic sprite drawn for every tree const too, so
    the trees a stroke plants or removes are real sprites in the oracle."""
    from descape import terrain_units, unit_sprites

    from test_unit_sprites import FILE_NAME

    entry = {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4, "mirroring_mode": 6, "frame_count": 1}
    mapping = {CONST: entry, **dict.fromkeys(terrain_units.TREE_CONSTS, entry)}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: mapping)
    unit_sprites.clear_caches()


def _fresh_cache(window, layers=None):
    """A new cache of the window's style over its current scenario, with its
    settings (`layers` in place of the window's, when given)."""
    from descape import render
    from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache

    mm = window.scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    kwargs = {
        "unit_filter": window._unit_filter, "sprites": window._sprites_enabled,
        "layers": window._layers if layers is None else layers,
    }
    if isinstance(window._cache, FlatChunkCache):
        return FlatChunkCache(window.scenario, tile_px, **kwargs)
    if isinstance(window._cache, IsoChunkCache):
        elevations, proj = render.elevations_and_proj(window.scenario)
        return IsoChunkCache(window.scenario, elevations, proj, tile_px, **kwargs)
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(window.scenario)
    return SlopedChunkCache(window.scenario, elevations, corner_rise, proj, tile_px, **kwargs)


def _ids_by_tile(by_tile: dict) -> dict:
    return {tile: [(id(u), color) for u, color in entries] for tile, entries in by_tile.items() if entries}


def _oracle_levels(window) -> tuple[int, ...]:
    """The visible level (mip 0, pinned) and a second resident one (1, else
    -1), then mip -1 too when both exist (A4's terrain eviction). Stepped and
    Sloped share one ladder since TASK-146, so both get (0, 1, -1): Sloped's
    -1 stays checked as the third level."""
    levels = window._cache.mip_levels()
    return (0, *(mip for mip in (1, -1) if mip in levels)) if len(levels) > 1 else (0,)


def _render_levels(window) -> None:
    """Every oracle level's whole canvas, all left resident (a chunk evicted
    in between would redraw fresh and hide a stale one)."""
    cache = window._cache
    for mip in _oracle_levels(window):
        cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)
    for mip in _oracle_levels(window):
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, 0, 0, *cache.canvas_dims(mip))
        resident = sum(1 for m, _cx, _cy in cache._cache if m == mip)
        assert resident == (cx1 - cx0 + 1) * (cy1 - cy0 + 1), f"mip {mip}: {resident} chunks resident"


def _check_against_fresh(window, step: str, numpy: bool = False) -> None:
    """units_by_tile (Flat 2D: its unit rows) equals a fresh build's; every
    oracle level's pixels equal a fresh cache's: on the active backend, or with
    `numpy` a numpy one, so native is not only checked against native
    (TASK-031.08/.09). The numpy reference replaces the native one, never
    both: a numpy fresh build costs several times a native one."""
    from descape import composite_backend, render
    from descape.render_cache import FlatChunkCache

    cache = window._cache
    if numpy and os.environ.get("DESCAPE_REQUIRE_NATIVE") == "1":
        assert composite_backend.active_backend() == "native", composite_backend.unavailable_reason
    live = {}
    for mip in _oracle_levels(window):
        live[mip] = cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)
    with composite_backend.use_backend("numpy" if numpy else composite_backend.active_backend()):
        fresh = _fresh_cache(window)
        if isinstance(cache, FlatChunkCache):
            for name, live_rows, fresh_rows in (
                ("bboxes", cache.unit_draws[0], fresh.unit_draws[0]),
                ("colors", cache.unit_draws[1], fresh.unit_draws[1]),
                ("row_uid", cache._row_uid, fresh._row_uid),
                ("row_player", cache._row_player, fresh._row_player),
            ):
                assert np.array_equal(live_rows, fresh_rows), f"{step}: Flat unit rows' {name} are stale"
        else:
            fresh_units = render._units_by_tile(window.scenario, cache.unit_filter)
            assert _ids_by_tile(cache.units_by_tile) == _ids_by_tile(fresh_units), f"{step}: units_by_tile is stale"
        for mip, pixels in live.items():
            assert np.array_equal(pixels, fresh.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)), (
                f"{step}: mip {mip} differs from a fresh {'numpy ' if numpy else ''}render"
            )


def _over_cap_window(style: str, monkeypatch, visible: int = 0):
    """_styled_window() with sprites on, the cap pinned under a stroke's trees,
    mip 0 shown (the whole canvas) and every oracle level resident. `visible`
    is VISIBLE_SPLICE_MAX_UNITS: 0 (the default) keeps the in-place and
    wholesale fallbacks, which stay the path past that bound."""
    monkeypatch.setattr(render_cache, "UNIT_SPLICE_MAX_UNITS", OVER_CAP)
    monkeypatch.setattr(render_cache, "VISIBLE_SPLICE_MAX_UNITS", visible)
    window = _styled_window(style)
    window.show_sprites_action.setChecked(True)
    assert window._cache.sprites_enabled
    window._cache.max_bytes *= 8  # room for every oracle level at once
    _cx0, _cy0, cx1, cy1 = window._cache.chunk_index_range(0, 0, 0, *window._cache.canvas_dims(0))
    monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (0, 0, 0, cx1, cy1))
    return window


def _over_cap_strokes(window, before_step=None) -> None:
    """Stroke A plants 8 oaks, stroke B replaces 4 of them with palms and adds
    4 more: both past OVER_CAP. Every oracle level is re-rendered right before
    each stroke, then checked against a fresh build, failing as `<step>: ...`."""
    for step, terrain, tiles in (("stroke A", FOREST_OAK, STROKE_TILES), ("stroke B", FOREST_PALM, SHIFTED_TILES)):
        _render_levels(window)
        if before_step is not None:
            before_step(step)
        _draw(window, terrain, tiles)
        _check_against_fresh(window, step)


@pytest.mark.parametrize("repaint", ["patch", "evict"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_over_cap_stroke_end_takes_the_scoped_path_and_matches_a_fresh_render(
    style, repaint, tree_sprites, monkeypatch
) -> None:
    """Past UNIT_SPLICE_MAX_UNITS the stroke end still hands its splices to
    the scoped repaint; the cache refuses with `cap` and updates units_by_tile
    in place. No whole-canvas eviction and no render._units_by_tile() walk."""
    from descape import render

    _pin_area_ratio(monkeypatch, 0 if repaint == "evict" else float("inf"))
    window = _over_cap_window(style, monkeypatch)
    try:
        calls = _spy(window, monkeypatch)
        evictions = _spy_evictions(window, monkeypatch)
        walks = []
        real_walk = render._units_by_tile
        monkeypatch.setattr(render, "_units_by_tile", lambda *a, **k: (walks.append(1), real_walk(*a, **k))[1])
        canvas = (0, 0, *window._cache.canvas_dims(0))
        for step, terrain, tiles in (("stroke A", FOREST_OAK, STROKE_TILES), ("stroke B", FOREST_PALM, SHIFTED_TILES)):
            _render_levels(window)
            del walks[:]
            _draw(window, terrain, tiles)
            changed = calls["changed"][-1]
            assert changed and len(changed) > OVER_CAP, f"{step}: took {changed}"
            assert walks == [], f"{step}: walked every unit"
            assert canvas not in evictions, f"{step}: evicted the whole canvas"
            assert bool(evictions) is (repaint == "evict"), f"{step}: {evictions}"
            del evictions[:]
        assert calls["in_place"] == 2 and calls["wholesale"] == 0
        monkeypatch.setattr(render, "_units_by_tile", real_walk)
        _check_against_fresh(window, "end")
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_over_cap_strokes_match_a_fresh_render_on_every_resident_level(style, tree_sprites, monkeypatch) -> None:
    window = _over_cap_window(style, monkeypatch)
    try:
        _over_cap_strokes(window)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", ["stroke A", "stroke B"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_over_cap_check_names_the_one_stroke_whose_update_went_stale(
    style, broken, tree_sprites, monkeypatch
) -> None:
    """Control (GOTCHAS: a step-k break fails naming step k): the in-place
    units_by_tile update is skipped on stroke `broken` only, claiming success."""
    window = _over_cap_window(style, monkeypatch)
    try:
        cache = window._cache
        real = cache._update_units_by_tile_in_place
        current = {}

        def update(changed):
            if current.get("step") == broken:
                cache._units_version += 1
                return True
            return real(changed)

        monkeypatch.setattr(cache, "_update_units_by_tile_in_place", update)
        with pytest.raises(AssertionError, match=f"{broken}: "):
            _over_cap_strokes(window, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


# Flat 2D has no area-ratio patch-or-evict choice, so only "patch" applies to it.
WITHIN_CAP_CASES = [
    (style, repaint) for style in ("Stepped", "Sloped", "FlatIso", "Flat") for repaint in ("patch", "evict")
    if not (style == "Flat" and repaint == "evict")
]


@pytest.mark.parametrize("style, repaint", WITHIN_CAP_CASES, ids=[f"{s}-{r}" for s, r in WITHIN_CAP_CASES])
def test_within_cap_strokes_their_undo_and_redo_match_a_fresh_render_on_every_resident_level(
    style, repaint, tree_sprites, monkeypatch
) -> None:
    """Plan Verification 4 (zoom and pan): the within-cap splice, on every resident level.
    - More than one level is resident, so a level the splice skipped would be checked.
    - Strokes A and B, B's undo and its redo each splice (non-empty `changed`, no wholesale rebuild).
    - Each step evicts at the visible level only when `repaint` is evict.
    - After each, units_by_tile (Flat 2D: its unit rows) and every oracle level's pixels equal a fresh
      cache's: a numpy one for stroke B, the undo and the redo (stroke A is setup, on the active backend)."""
    _pin_area_ratio(monkeypatch, 0 if repaint == "evict" else float("inf"))
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "_SPLICE_COST_RATIO", 0)
    window = _styled_window(style)
    try:
        window.show_sprites_action.setChecked(True)
        assert window._cache.sprites_enabled
        window._cache.max_bytes *= 8  # room for every oracle level at once
        _cx0, _cy0, cx1, cy1 = window._cache.chunk_index_range(0, 0, 0, *window._cache.canvas_dims(0))
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (0, 0, 0, cx1, cy1))
        assert len(_oracle_levels(window)) > 1
        calls = _spy(window, monkeypatch)
        evictions = _spy_evictions(window, monkeypatch)
        steps = (
            ("stroke A", lambda: _draw(window, FOREST_OAK)),
            ("stroke B", lambda: _draw(window, FOREST_PALM, SHIFTED_TILES)),
            ("undo B", window.undo),
            ("redo B", window.redo),
        )
        for step, act in steps:
            _render_levels(window)
            act()
            changed = calls["changed"][-1]
            assert changed and len(changed) <= render_cache.UNIT_SPLICE_MAX_UNITS, f"{step}: took {changed}"
            assert calls["wholesale"] == 0, f"{step}: wholesale source rebuild"
            assert bool(evictions) is (repaint == "evict"), f"{step}: {evictions}"
            del evictions[:]
            _check_against_fresh(window, step, numpy=step != "stroke A")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_over_cap_flat_stroke_end_stays_wholesale(monkeypatch) -> None:
    """Flat keeps the cap: with its cost ratio off, only the cap sends it wholesale."""
    monkeypatch.setattr(render_cache, "UNIT_SPLICE_MAX_UNITS", OVER_CAP)
    _splice_any_batch(monkeypatch)
    window = _styled_window("Flat")
    try:
        calls = _spy(window, monkeypatch)
        _draw(window, FOREST_OAK)
        assert calls["changed"] == [None]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", ["stroke A", "stroke B"])
def test_the_over_cap_check_names_the_stroke_that_left_a_second_level_stale(
    broken, tree_sprites, monkeypatch
) -> None:
    """Control for the second resident level: on stroke `broken` only, the
    eviction of the non-visible levels is dropped, and the check must fail on
    that level, naming the stroke."""
    window = _over_cap_window("Stepped", monkeypatch)
    try:
        cache = window._cache
        second = _oracle_levels(window)[1]
        real_evict = cache.invalidate_region
        current = {}

        def evict(bbox, levels=None, **kwargs):
            if current.get("step") == broken and levels is not None and 0 not in levels:
                return None
            return real_evict(bbox, levels=levels, **kwargs)

        monkeypatch.setattr(cache, "invalidate_region", evict)
        with pytest.raises(AssertionError, match=f"{broken}: mip {second} differs"):
            _over_cap_strokes(window, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- Over-cap membership undo/redo (TASK-031.61 A3) ---

# Starts left of STROKE_TILES, so its undo puts A's first 4 oaks back mid-list (not tail adds).
EARLIER_TILES = [(16 + i, 30) for i in range(8)]
HISTORY_STEPS = ("undo B", "undo A", "redo A", "redo B")


def _enter_units(window) -> None:
    from PyQt5.QtWidgets import QApplication

    window.mode_combo.setCurrentText("Units")
    QApplication.processEvents()
    assert window.mode == "units"


def _check_index_and_selection(window, step: str, selected: list) -> list:
    """The pick index equals a fresh build, and the selection is the previous
    one minus the keys a fresh index no longer resolves, in order (a dropped
    key never comes back). Returns that selection."""
    from descape import unit_pick

    index = window.map_view._unit_index
    fresh = unit_pick.build_index(window.scenario, window._unit_filter)
    assert index is not None and set(index.by_key) == set(fresh.by_key), f"{step}: the pick index is stale"
    assert index.by_tile == fresh.by_tile, f"{step}: the pick index is stale"
    expected = [k for k in selected if fresh.entry_for_key(k) is not None]
    assert window._selection == expected, f"{step}: selection"
    return expected


def _over_cap_history(window, before_step=None, after_step=None) -> None:
    """Strokes A (8 oaks) and B (8 palms over A's first 4 oaks), both past
    OVER_CAP, then HISTORY_STEPS. Every oracle level is re-rendered right
    before each step, then checked against a fresh build, failing as `<step>: ...`."""
    _draw(window, FOREST_OAK, STROKE_TILES)
    _draw(window, FOREST_PALM, EARLIER_TILES)
    acts = {"undo": window.undo, "redo": window.redo}
    for step in HISTORY_STEPS:
        _render_levels(window)
        if before_step is not None:
            before_step(step)
        acts[step.split()[0]]()
        _check_against_fresh(window, step)
        if after_step is not None:
            after_step(step)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_undo_and_redo_of_over_cap_draws_keep_their_splices_and_match_a_fresh_render(
    style, tree_sprites, monkeypatch
) -> None:
    """Past UNIT_SPLICE_MAX_UNITS a Draw's undo and redo still hand the scoped
    repaint their splices (never the whole canvas); the cache picks in-place
    or a units_by_tile rebuild. The pick index and selection follow, in Units mode."""
    window = _over_cap_window(style, monkeypatch)
    try:
        calls = _spy(window, monkeypatch)
        evictions = _spy_evictions(window, monkeypatch)
        canvas = (0, 0, *window._cache.canvas_dims(0))
        state = {}

        def before(step):
            if step == HISTORY_STEPS[0]:
                _enter_units(window)
                window._selection = [(0, u.reference_id) for u in window.scenario.unit_manager.units[0]]
                window._refresh_selection_view()
                state["selected"] = list(window._selection)
                assert len(state["selected"]) >= 12
            del evictions[:]

        def after(step):
            changed = calls["changed"][-1]
            assert changed and len(changed) > OVER_CAP, f"{step}: took {changed}"
            assert canvas not in evictions, f"{step}: evicted the whole canvas"
            state["selected"] = _check_index_and_selection(window, step, state["selected"])

        _over_cap_history(window, before, after)
        # undo B re-inserts mid-list (a units_by_tile rebuild); the other three update in place.
        assert calls["in_place"] == 2 + 3, calls
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", ["undo B", "redo B"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_over_cap_history_check_names_the_one_step_whose_units_went_stale(
    style, broken, tree_sprites, monkeypatch
) -> None:
    """Control (GOTCHAS: a step-k break fails naming step k): invalidate_units
    does nothing on step `broken` only (a rebuild on undo B, in place on redo B)."""
    window = _over_cap_window(style, monkeypatch)
    try:
        cache = window._cache
        real = cache.invalidate_units
        current = {}

        def invalidate(changed=None, splice_levels=True, **kwargs):
            if current.get("step") == broken:
                return None
            return real(changed, splice_levels=splice_levels, **kwargs)

        monkeypatch.setattr(cache, "invalidate_units", invalidate)
        with pytest.raises(AssertionError, match=f"{broken}: "):
            _over_cap_history(window, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped", "Flat"])
def test_the_gaia_membership_diff_is_uncapped_except_on_flat(style, monkeypatch) -> None:
    monkeypatch.setattr(render_cache, "UNIT_SPLICE_MAX_UNITS", OVER_CAP)
    _splice_any_batch(monkeypatch)
    window = _styled_window(style)
    try:
        _draw(window, FOREST_OAK)
        calls = _spy(window, monkeypatch)
        diff = window._membership_diff(window.edit_history.peek_undo(), undo=True)
        window.undo()
        if style == "Flat":
            assert diff is None and calls["changed"] == [None]
        else:
            assert diff is not None and len(diff[0]) == len(STROKE_TILES)
            assert calls["changed"][-1] and len(calls["changed"][-1]) > OVER_CAP
            assert np.array_equal(_canvas(window), _fresh(window, style))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_undo_and_redo_of_an_over_cap_two_player_place_keep_their_splices(style, tree_sprites, monkeypatch) -> None:
    """A3's _if_spliceable(cap=None) widens the paste shape too: one Place
    record of 6 units across players 1 and 2 (past OVER_CAP) undoes and redoes
    through its splices, in place, matching a fresh build."""
    window = _over_cap_window(style, monkeypatch)
    try:
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Place", [1, 2]):
            for i in range(3):
                model.add(1, CONST, 40.5 + 3 * i, 40.5)
                model.add(2, CONST, 40.5 + 3 * i, 46.5)
        calls = _spy(window, monkeypatch)
        for step, act in (("undo", window.undo), ("redo", window.redo)):
            _render_levels(window)
            act()
            changed = calls["changed"][-1]
            assert changed and len(changed) == 6, f"{step}: took {changed}"
            assert {s.player_id for s in changed} == {1, 2}, step
            _check_against_fresh(window, step)
        assert calls["in_place"] == 2 and calls["wholesale"] == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- A terrain + unit history move composites once (TASK-031.61 A4) ---


def _spy_composite_order(window, monkeypatch) -> list:
    """Each cache patch(), patch_sources() and patch_composite() call, and each
    unit step (_patch_unit_edit_cache), in order. patch() calls the two halves
    through the instance, so its own halves are recorded after it."""
    events = []
    cache = window._cache
    for name in ("patch", "patch_sources", "patch_composite"):
        real = getattr(cache, name)
        monkeypatch.setattr(cache, name, lambda *a, _n=name, _r=real, **k: (events.append(_n), _r(*a, **k))[1])
    real_step = window._patch_unit_edit_cache
    monkeypatch.setattr(
        window, "_patch_unit_edit_cache", lambda *a, **k: (events.append("unit_step"), real_step(*a, **k))[1]
    )
    return events


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_draw_undo_and_redo_refresh_terrain_sources_only_before_the_unit_step(
    style, tree_sprites, monkeypatch
) -> None:
    """Undo 1's shape: the tile patch runs only its source half, and the unit
    step composites the terrain with its own bbox, so nothing is composited
    first and evicted after. Every oracle level matches a fresh build."""
    window = _over_cap_window(style, monkeypatch)
    try:
        _draw(window, FOREST_OAK, STROKE_TILES)
        events = _spy_composite_order(window, monkeypatch)
        for step, act in (("undo", window.undo), ("redo", window.redo)):
            _render_levels(window)
            del events[:]
            act()
            assert "unit_step" in events, f"{step}: {events}"
            assert events[: events.index("unit_step")] == ["patch_sources"], f"{step}: {events}"
            _check_against_fresh(window, step)
    finally:
        window.edit_history.mark_saved()
        window.close()


def _blocked_stroke_history(window, monkeypatch, before_step=None, block=None) -> None:
    """A stroke over a 3x3 block the planner treats as a cliff's footprint
    (terrain, no tree: GH #183; no cliff unit, whose mark would hide the
    terrain) and STROKE_TILES (8 oaks), so the terrain bbox reaches well past
    the unit one; then its undo and redo, each checked as `<step>: ...`.
    `block`: the tiles drawn inside that footprint, instead of all nine."""
    from types import SimpleNamespace

    from descape import cliff_chain

    if block is None:
        block = [(x, y) for y in range(10, 13) for x in range(10, 13)]
    monkeypatch.setattr(cliff_chain, "existing_cliffs", lambda *_a: [SimpleNamespace(bounds=(10, 13, 10, 13))])
    _draw(window, FOREST_OAK, [*block, *STROKE_TILES])
    assert len(window.scenario.unit_manager.units[0]) == len(STROKE_TILES)
    for step, act in (("undo", window.undo), ("redo", window.redo)):
        _render_levels(window)
        if before_step is not None:
            before_step(step)
        act()
        _check_against_fresh(window, step)


@pytest.mark.parametrize("repaint", ["patch", "evict", "rects", "no_target"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_history_moves_terrain_past_its_units_is_repainted_with_them(style, repaint, tree_sprites, monkeypatch) -> None:
    """Both styles check mip 0 (visible), 1 and -1, all resident (Sloped's
    finer level since TASK-146). `rects` is Stepped's per-chunk patch (C1),
    which sizes the terrain from the move's tiles."""
    _pin_area_ratio(
        monkeypatch, 0 if repaint in ("evict", "rects") else float("inf"),
        rects=float("inf") if repaint == "rects" else 0,
    )
    window = _over_cap_window(style, monkeypatch)
    try:
        assert _oracle_levels(window) == (0, 1, -1)
        if repaint == "no_target":
            monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: None)
        _blocked_stroke_history(window, monkeypatch)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", ["undo", "redo"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_blocked_stroke_check_names_the_step_whose_terrain_bbox_was_dropped(
    style, broken, tree_sprites, monkeypatch
) -> None:
    """Control: on step `broken` only, the unit step's repaint drops the
    deferred terrain bbox, and the check must fail there."""
    _pin_area_ratio(monkeypatch, float("inf"))
    window = _over_cap_window(style, monkeypatch)
    try:
        real = window._repaint_unit_edit_split
        current = {}

        def split(*args, terrain_bbox=None, **kwargs):
            return real(*args, terrain_bbox=None if current.get("step") == broken else terrain_bbox, **kwargs)

        monkeypatch.setattr(window, "_repaint_unit_edit_split", split)
        with pytest.raises(AssertionError, match=f"{broken}: "):
            _blocked_stroke_history(window, monkeypatch, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("splice", ["empty", "no_tiles"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_unit_step_that_repaints_nothing_still_composites_the_deferred_terrain(style, splice, monkeypatch) -> None:
    """The unit step's early returns (no splices, or splices touching no tile:
    here a no-op move of an off-map unit) composite the terrain bbox
    _apply_dirty() deferred, as its patch would have."""
    from descape.render_cache import UnitSplice

    window = _over_cap_window(style, monkeypatch)
    try:
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [1]):
            unit = model.add(1, CONST, -5.5, -5.5)
        own, tiles = window._unit_footprint(unit)
        assert tiles == ()
        _render_levels(window)
        mm = window.scenario.map_manager
        dirty = [y * mm.map_width + x for x, y in STROKE_TILES]
        for i in dirty:
            mm.terrain[i].terrain_id = FOREST_PALM
        deferred: list = []
        assert window._apply_dirty(dirty, deferred=deferred) is False and len(deferred) == 1
        changed = [] if splice == "empty" else [UnitSplice(1, 0, unit, own, own, (), ())]
        window._after_unit_mutation(changed, batch=True, terrain_bbox=deferred[0])
        _check_against_fresh(window, splice)
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- The unit split skips the terrain re-read where the mirrors are current (TASK-031.61 C0) ---

BLENDED_STEPS = ("stroke", "undo", "redo")


def _spy_split_rereads(window, monkeypatch) -> dict:
    """Counts _repaint_unit_edit_split() calls, and records the bbox of every
    terrain mirror re-read (_refresh_terrain_ids) made inside one."""
    seen = {"splits": 0, "rereads": []}
    inside = [False]
    real_split, real_reread = window._repaint_unit_edit_split, window._cache._refresh_terrain_ids

    def split(*args, **kwargs):
        seen["splits"] += 1
        inside[0] = True
        try:
            return real_split(*args, **kwargs)
        finally:
            inside[0] = False

    def reread(bbox):
        if inside[0]:
            seen["rereads"].append(bbox)
        return real_reread(bbox)

    monkeypatch.setattr(window, "_repaint_unit_edit_split", split)
    monkeypatch.setattr(window._cache, "_refresh_terrain_ids", reread)
    return seen


def _place_one(window, x: float = 40.5, y: float = 40.5) -> None:
    """A single-unit Place through _unit_edit(), with its splice."""
    from descape.render_cache import UnitSplice

    model = window._ensure_unit_edits()
    splices: list = []
    with window._unit_edit(model, "Place unit", [1], splices=splices):
        unit = model.add(1, CONST, x, y)
        own, tiles = window._unit_footprint(unit)
        splices.append(UnitSplice(1, len(window.scenario.unit_manager.units[1]) - 1, unit, None, own, (), tiles))


@pytest.mark.parametrize("repaint", ["patch", "evict"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_unit_split_skips_the_terrain_re_read_where_the_mirrors_are_current(
    style, repaint, tree_sprites, monkeypatch
) -> None:
    """A Draw stroke end (its drag re-read the mirrors step by step), its undo
    and redo (A4's patch_sources() just did) and a _unit_edit() (no tile
    written) evict without re-reading. Only an eager patch of the visible
    level still re-reads, once, as patch() always does."""
    _pin_area_ratio(monkeypatch, 0 if repaint == "evict" else float("inf"))
    window = _over_cap_window(style, monkeypatch)
    try:
        seen = _spy_split_rereads(window, monkeypatch)
        patched = {"evict": 0, "patch": 1}[repaint]
        steps = (
            ("stroke end", lambda: _draw(window, FOREST_OAK, STROKE_TILES), patched),
            ("undo", window.undo, patched),
            ("redo", window.redo, patched),
            ("unit edit", lambda: _place_one(window), 1),  # a single-unit tool always patches
        )
        for step, act, own in steps:
            _render_levels(window)  # a second level resident, so both evictions run
            assert len(window._cache.resident_levels()) > 1
            seen["splits"] = 0
            del seen["rereads"][:]
            act()
            assert seen["splits"] == 1, f"{step}: the split ran {seen['splits']} times"
            assert len(seen["rereads"]) == own, f"{step}: the split re-read the terrain mirrors over {seen['rereads']}"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.fixture
def blended_trees(tree_sprites, blend_install):  # noqa: F811
    """tree_sprites plus the imported blend_install: test_terrain_blend_render's
    synthetic blendomatic, blend table and layer mask (pytest hides the
    install). Its table holds FOREST_OAK, FOREST_PALM and LAYER_ID."""


def _blended_window(style: str, monkeypatch, repaint: str = "evict", visible: int = 0):
    """_over_cap_window() with terrain blending effective (the tests request
    blended_trees) and the batch repaint pinned. Evict is the path that skips
    the re-read, so a stale mirror would be composited as it stands. `rects`
    is Stepped's per-chunk patch in place of that eviction (TASK-031.61 C1);
    on Sloped it pins the same thing as `evict`."""
    from descape import render

    _pin_area_ratio(
        monkeypatch, float("inf") if repaint == "patch" else 0, rects=float("inf") if repaint == "rects" else 0
    )
    window = _over_cap_window(style, monkeypatch, visible)
    assert render.blend_effective(window._cache.layers), "blending is off, so the case would prove nothing"
    return window


def _blended_history(window, before_step=None) -> None:
    """Over grass with 8 palms on EARLIER_TILES: a Draw stroke that plants 8
    oaks over 4 of them (past OVER_CAP), its undo and its redo. Every oracle
    level is re-rendered right before each step, then checked against a fresh
    build, failing as `<step>: ...`."""
    _draw(window, FOREST_PALM, EARLIER_TILES)
    acts = {"stroke": lambda: _draw(window, FOREST_OAK, STROKE_TILES), "undo": window.undo, "redo": window.redo}
    for step in BLENDED_STEPS:
        _render_levels(window)
        if before_step is not None:
            before_step(step)
        acts[step]()
        _check_against_fresh(window, step)


@pytest.mark.parametrize("repaint", ["patch", "evict", "rects"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_blended_over_cap_draw_its_undo_and_redo_match_a_fresh_render_on_every_resident_level(
    style, repaint, blended_trees, monkeypatch
) -> None:
    """Sprites and blending on: the visible level (mip 0) and the second
    resident one(s) equal a fresh build after each step, the split having
    re-read no terrain. `rects` patches per-chunk rects on Stepped (C1), once
    per step (the setup stroke, with no chunk resident yet, takes the reach
    fallback). Sloped never does."""
    from dataclasses import replace

    from descape import render

    window = _blended_window(style, monkeypatch, repaint)
    try:
        assert render.blend_effective(window._cache.layers)
        assert len(_oracle_levels(window)) > 1
        rect_patches = _spy_rect_patches(window, monkeypatch)
        _blended_history(window)
        assert len(rect_patches) == (3 if (style, repaint) == ("Stepped", "rects") else 0), rect_patches
        cache = window._cache
        unblended = _fresh_cache(window, layers=replace(window._layers, terrain_blend=False))
        w, h = cache.canvas_dims(0)
        assert not np.array_equal(cache.render_rect(0, 0, w, h, mip=0), unblended.render_rect(0, 0, w, h, mip=0)), (
            "blending changed no pixel, so the checks above proved nothing about it"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", BLENDED_STEPS)
@pytest.mark.parametrize(("style", "repaint"), [("Stepped", "evict"), ("Stepped", "rects"), ("Sloped", "evict")])
def test_the_blended_check_names_the_one_step_whose_terrain_mirrors_went_stale(
    style, repaint, broken, blended_trees, monkeypatch
) -> None:
    """Control (GOTCHAS: a step-k break fails naming step k): on step `broken`
    only, nothing re-reads the terrain mirrors, which is the state the split
    now trusts its three callers never to hand it."""
    from descape import render

    window = _blended_window(style, monkeypatch, repaint)
    try:
        assert render.blend_effective(window._cache.layers)
        cache = window._cache
        real = cache._refresh_terrain_ids
        current = {}
        monkeypatch.setattr(
            cache, "_refresh_terrain_ids", lambda bbox: None if current.get("step") == broken else real(bbox)
        )
        with pytest.raises(AssertionError, match=f"{broken}: mip 0 differs"):
            _blended_history(window, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", BLENDED_STEPS)
@pytest.mark.parametrize("repaint", ["evict", "rects"])
def test_the_blended_check_names_the_step_that_left_a_second_level_stale(
    repaint, broken, blended_trees, monkeypatch
) -> None:
    """Control for the second resident level, blending on: on step `broken`
    only, the eviction of the non-visible levels is dropped, and the check
    must fail on that level, naming the step. Under `rects` too (C1), where
    only the visible level is patched in place."""
    from descape import render

    window = _blended_window("Stepped", monkeypatch, repaint)
    try:
        assert render.blend_effective(window._cache.layers)
        cache = window._cache
        second = _oracle_levels(window)[1]
        real_evict = cache.invalidate_region
        current = {}

        def evict(bbox, levels=None, **kwargs):
            if current.get("step") == broken and levels is not None and 0 not in levels:
                return None
            return real_evict(bbox, levels=levels, **kwargs)

        monkeypatch.setattr(cache, "invalidate_region", evict)
        with pytest.raises(AssertionError, match=f"{broken}: mip {second} differs"):
            _blended_history(window, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


def _spy_key_stash(window, monkeypatch) -> dict:
    """Counts, over the window cache's key map lookups, the tiles found
    UNKNOWN and the tiles derived; the rest came from its stash."""
    from descape import terrain_blend

    key_map = window._cache._blend_key_map
    real = key_map.lookup
    seen = {"unknown": 0, "derived": 0}

    def lookup(idx, derive):
        seen["unknown"] += int(np.unique(idx[key_map.ids[idx] == terrain_blend.UNKNOWN]).size)

        def counted(xs, ys):
            seen["derived"] += len(xs)
            return derive(xs, ys)

        return real(idx, counted)

    monkeypatch.setattr(key_map, "lookup", lookup)
    return seen


@pytest.mark.parametrize("style", ["Stepped", "Sloped", "Flat"])
def test_a_blended_draw_undo_redo_undo_reuses_stashed_keys_and_matches_a_fresh_render(
    style, blended_trees, monkeypatch
) -> None:
    """TASK-031.69 T7: sprites and blending on, a Draw stroke over palms, its
    undo, redo and undo again. After each, every oracle level equals a fresh
    build. The undos and the redo take keys from the stash: the first undo
    some, and the second undo every one, since only the redo's states were
    stored since the first."""
    from descape import render

    window = _blended_window(style, monkeypatch, "patch")
    try:
        assert render.blend_effective(window._cache.layers)
        _draw(window, FOREST_PALM, EARLIER_TILES)
        seen = _spy_key_stash(window, monkeypatch)
        steps = (
            ("stroke", lambda: _draw(window, FOREST_OAK, STROKE_TILES)),
            ("undo", window.undo),
            ("redo", window.redo),
            ("undo 2", window.undo),
        )
        counts = {}
        for step, act in steps:
            _render_levels(window)
            seen.update(unknown=0, derived=0)
            act()
            _check_against_fresh(window, step)
            counts[step] = dict(seen)
        assert counts["stroke"]["derived"] > 0, counts
        for step in ("undo", "redo", "undo 2"):
            assert counts[step]["unknown"] > counts[step]["derived"], f"{step}: no key came from the stash {counts}"
        assert counts["undo 2"]["unknown"] > 0 and counts["undo 2"]["derived"] == 0, counts
    finally:
        window.edit_history.mark_saved()
        window.close()


# Two of STROKE_TILES' oaks: each gets a layer, then the Eraser takes the oak and the layer.
ERASED_FIRST, ERASED_LAST = STROKE_TILES[2], STROKE_TILES[5]
# Grass with no unit near it, far enough from STROKE_TILES that no erased oak's bbox or rect reaches it.
BARE_LAYERED = (70, 80)


def _open_erase_stroke(window, **targets) -> None:
    """The Eraser, brush 1, with only `targets` ticked, pressed (no tile yet)."""
    window._on_tool_selected("eraser")
    window.brush_size_spin.setValue(1)
    for name, action in window.eraser_target_actions.items():
        action.setChecked(targets.get(name, False))
    window.on_edit_stroke_start()
    assert window._erase is not None


def _overlay_erase(window, seen: dict | None = None) -> None:
    """8 oaks, two of them on tiles given a layer (through the viewer's own
    funnel), then one Eraser drag with Trees and Terrain Overlay ticked: the
    first oak and its layer, the mid-stroke flush, the second, the release.
    Every oracle level is checked against a fresh build after the flush and
    after the release. `seen` (_spy_split_rereads) is checked at both too."""
    _draw(window, FOREST_OAK, STROKE_TILES)
    mm = window.scenario.map_manager
    layered = [y * mm.map_width + x for x, y in (ERASED_FIRST, ERASED_LAST)]
    for i in layered:
        mm.terrain[i].layer = LAYER_ID
    window._apply_dirty(layered)
    _render_levels(window)
    _check_against_fresh(window, "layered")
    trees = len(window.scenario.unit_manager.units[0])

    def expect_rereads(step: str) -> None:
        if seen is not None:
            # One per eviction: the other resident levels', then the visible one's.
            assert seen["splits"] == 1 and len(seen["rereads"]) == 2, f"{step}: {seen}"
            seen["splits"] = 0
            del seen["rereads"][:]

    _open_erase_stroke(window, trees=True, overlay=True)
    assert window._erase.overlay, "Terrain Overlay does not apply to this file"
    if seen is not None:
        seen["splits"] = 0
        del seen["rereads"][:]
    window.on_edit_stroke_tiles([ERASED_FIRST], 0)
    assert window._erase_refresh_timer.isActive()
    window._erase_refresh_timer.stop()
    window._flush_erase_refresh()
    assert window._erase is not None  # still mid-stroke
    assert mm.terrain[layered[0]].layer == -1 and len(window.scenario.unit_manager.units[0]) == trees - 1
    _check_against_fresh(window, "erase flush")
    expect_rereads("erase flush")
    _render_levels(window)
    window.on_edit_stroke_tiles([ERASED_LAST], 0)
    window._erase_refresh_timer.stop()
    window.on_edit_stroke_end()
    assert mm.terrain[layered[1]].layer == -1 and len(window.scenario.unit_manager.units[0]) == trees - 2
    _check_against_fresh(window, "erase end")
    expect_rereads("erase end")


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_terrain_overlay_erase_still_re_reads_and_repaints_the_layer_mid_stroke(
    style, blended_trees, monkeypatch
) -> None:
    """The case C0 leaves alone: Terrain Overlay writes `layer = -1` live, and
    the mirrors only catch up at a flush or the stroke end. Both Erase calls
    keep the split's re-read, and the flush repaints the cleared layer."""
    from descape import render

    window = _blended_window(style, monkeypatch)
    try:
        assert render.blend_effective(window._cache.layers)
        _overlay_erase(window, _spy_split_rereads(window, monkeypatch))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_overlay_erase_check_fails_if_the_flush_neither_applies_nor_re_reads_the_cleared_layer(
    style, blended_trees, monkeypatch
) -> None:
    """Control: mid-stroke the cleared tiles are not put through
    _apply_dirty() (the flush as it was), and the flush (the one defer_index
    caller here) passes terrain_current=True, so its evictions composite the
    stale layer mirror."""
    from descape import render

    window = _blended_window(style, monkeypatch)
    try:
        assert render.blend_effective(window._cache.layers)
        real_after = window._after_unit_mutation

        def after(changed=None, **kwargs):
            if kwargs.get("defer_index"):
                kwargs["terrain_current"] = True
            return real_after(changed, **kwargs)

        real_dirty = window._apply_dirty
        monkeypatch.setattr(
            window, "_apply_dirty", lambda tiles, **k: False if window._erase is not None else real_dirty(tiles, **k)
        )
        monkeypatch.setattr(window, "_after_unit_mutation", after)
        with pytest.raises(AssertionError, match="erase flush: mip 0 differs"):
            _overlay_erase(window)
    finally:
        window.edit_history.mark_saved()
        window.close()


def _erase_past_a_bare_layer(window, before_flush=None) -> None:
    """8 oaks and, well off them, a bare tile given a layer. One Eraser drag
    with Trees and Terrain Overlay ticked clears that layer (no unit there,
    so no flush is due), then takes an oak, flushes, and ends. Every oracle
    level is checked against a fresh build after the flush and the release."""
    _draw(window, FOREST_OAK, STROKE_TILES)
    mm = window.scenario.map_manager
    bare = BARE_LAYERED[1] * mm.map_width + BARE_LAYERED[0]
    mm.terrain[bare].layer = LAYER_ID
    window._apply_dirty([bare])
    _render_levels(window)
    _check_against_fresh(window, "layered")
    _open_erase_stroke(window, trees=True, overlay=True)
    assert window._erase.overlay, "Terrain Overlay does not apply to this file"
    window.on_edit_stroke_tiles([BARE_LAYERED], 0)
    assert mm.terrain[bare].layer == -1 and not window._erase_refresh_timer.isActive()
    window.on_edit_stroke_tiles([ERASED_FIRST], 0)
    assert window._erase_refresh_timer.isActive()
    window._erase_refresh_timer.stop()
    if before_flush is not None:
        before_flush()
    window._flush_erase_refresh()
    assert window._erase is not None  # still mid-stroke
    _check_against_fresh(window, "erase flush")
    window.on_edit_stroke_end()
    _check_against_fresh(window, "erase end")


@pytest.mark.parametrize("repaint", ["patch", "evict", "rects"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_erase_flush_repaints_a_cleared_layer_no_erased_unit_stood_on(
    style, repaint, blended_trees, monkeypatch
) -> None:
    """The flush's unit step repaints what its erased units reached, by bbox
    or by rect. A layer cleared on a tile outside that shows mid-drag only
    because the flush also puts the tiles cleared since the last one through
    _apply_dirty()."""
    window = _blended_window(style, monkeypatch, repaint)
    try:
        _erase_past_a_bare_layer(window)
        assert window._erase is None and window.edit_history.can_undo
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("repaint", ["evict", "rects"])
def test_the_bare_layer_check_fails_if_the_flush_drops_its_cleared_tiles(repaint, blended_trees, monkeypatch) -> None:
    """Control: the flush's cleared tiles are forgotten just before it runs,
    which is the flush as it was. The release still repairs the tile, so only
    the mid-stroke check can fail."""
    window = _blended_window("Stepped", monkeypatch, repaint)
    try:
        with pytest.raises(AssertionError, match="erase flush: mip 0 differs"):
            _erase_past_a_bare_layer(window, before_flush=lambda: window._erase.pending_layer_tiles.clear())  # noqa: PLW0108 -- _erase is read at the flush
    finally:
        window.on_edit_stroke_end()
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("act", ["undo", "unit edit"])
def test_inside_an_open_erase_stroke_the_split_keeps_its_re_read(act, tree_sprites, monkeypatch) -> None:
    """Nothing refuses a keyboard undo or unit edit during an Eraser drag, and
    the drag's Terrain Overlay writes are not in the mirrors yet, so neither
    caller vouches for them until the stroke has ended."""
    _pin_area_ratio(monkeypatch, 0)
    window = _over_cap_window("Stepped", monkeypatch)
    try:
        _draw(window, FOREST_OAK, STROKE_TILES)
        _render_levels(window)
        _open_erase_stroke(window, overlay=True)
        seen = _spy_split_rereads(window, monkeypatch)
        if act == "undo":
            window.undo()
        else:
            _place_one(window)
        # The other levels' eviction, then the visible level's (undo) or its eager patch (unit edit).
        assert seen["splits"] == 1 and len(seen["rereads"]) == 2, seen
        window.on_edit_stroke_end()
        assert window._erase is None
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- Stepped patches one rect per chunk instead of evicting (TASK-031.61 C1) ---

# A tile row, so a diagonal on screen: five 16x16-tile blocks whose rects step down across several
# mip-0 chunks and together cover far less than their union bbox.
LONG_TILES = [(10 + i, 40) for i in range(60)]
# The same row 20 tiles on: palms replace 40 of LONG_TILES' oaks and 20 more land on grass.
LONG_SHIFTED = [(30 + i, 40) for i in range(60)]
LONG_STEPS = ("stroke A", "stroke B", "undo B", "redo B")
# One tile wide, inside the footprint _blocked_stroke_history() plants no tree on.
EDGE_TILES = [(11, 10), (11, 11), (11, 12)]


def _spy_rect_patches(window, monkeypatch) -> list:
    """Every cache.patch_composite_rects() call from here on, as a dict: its
    `rects`, `levels`, `terrain_current`, and `composites`, the
    (mip, x0, y0, x1, y1) of each _composite_rect() made inside it."""
    calls: list = []
    cache = window._cache
    real_rects, real_composite = cache.patch_composite_rects, cache._composite_rect
    inside: list = []

    def patch_rects(rects, levels=None, rebuild_levels=None, terrain_current=False):
        call = {"rects": list(rects), "levels": levels, "terrain_current": terrain_current, "composites": []}
        calls.append(call)
        inside.append(call)
        try:
            return real_rects(rects, levels=levels, rebuild_levels=rebuild_levels, terrain_current=terrain_current)
        finally:
            inside.pop()

    def composite(mip, x0, y0, x1, y1):
        if inside:
            inside[-1]["composites"].append((mip, x0, y0, x1, y1))
        return real_composite(mip, x0, y0, x1, y1)

    monkeypatch.setattr(cache, "patch_composite_rects", patch_rects)
    monkeypatch.setattr(cache, "_composite_rect", composite)
    return calls


def _spy_all_evictions(window, monkeypatch) -> list:
    """Every cache.invalidate_region() from here on, as (bbox, levels)."""
    evictions = []
    real_evict = window._cache.invalidate_region

    def evict(bbox, levels=None, **kwargs):
        evictions.append((bbox, None if levels is None else tuple(levels)))
        return real_evict(bbox, levels=levels, **kwargs)

    monkeypatch.setattr(window._cache, "invalidate_region", evict)
    return evictions


def _spy_qt_regions(window, monkeypatch) -> list:
    """Every bbox the canvas is asked to repaint (map_view.invalidate_region)."""
    regions = []
    real = window.map_view.invalidate_region
    monkeypatch.setattr(window.map_view, "invalidate_region", lambda bbox: (regions.append(tuple(bbox)), real(bbox))[1])
    return regions


def _seed_planner(monkeypatch) -> None:
    """Every stroke plants the same trees, so two windows can be compared."""
    import random

    from descape import terrain_units

    real_plan = terrain_units.plan_terrain_units
    monkeypatch.setattr(
        terrain_units, "plan_terrain_units", lambda *a, **k: real_plan(*a, **{**k, "rng": random.Random(7)})
    )


def _long_batches(window, each) -> None:
    """LONG_STEPS: 60 oaks along LONG_TILES, 60 palms along LONG_SHIFTED, then
    that stroke's undo and redo. Every oracle level is re-rendered right
    before each step; each(step, act) runs it."""
    acts = (
        lambda: _draw(window, FOREST_OAK, LONG_TILES),
        lambda: _draw(window, FOREST_PALM, LONG_SHIFTED),
        window.undo,
        window.redo,
    )
    for step, act in zip(LONG_STEPS, acts, strict=True):
        _render_levels(window)
        each(step, act)


def _inside(outer, inner) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _mip0_chunks(window) -> set:
    return {key for key in window._cache._cache if key[0] == 0}


@pytest.mark.parametrize("style", ["Stepped", "FlatIso", "Sloped"])
def test_an_over_ratio_batch_patches_one_rect_per_chunk_and_evicts_nothing_at_the_visible_level(
    style, blended_trees, monkeypatch
) -> None:
    """A Draw stroke end, a second one over it, its undo and its redo, each
    past the area ratio and past UNIT_SPLICE_MAX_UNITS, sprites and blending
    on. Stepped (and Flat + Isometric View, the same cache) evicts no chunk of
    the visible level: it composites one rect per chunk the edit reaches, all
    of them inside the bbox Qt repaints and together smaller than it. The
    other resident levels are still evicted, and every level matches a fresh
    build. Sloped keeps evicting."""
    from descape import render

    window = _blended_window(style, monkeypatch, "rects")
    try:
        cache = window._cache
        assert render.blend_effective(cache.layers)
        rect_patches = _spy_rect_patches(window, monkeypatch)
        evictions = _spy_all_evictions(window, monkeypatch)
        regions = _spy_qt_regions(window, monkeypatch)
        others = tuple(mip for mip in sorted(_oracle_levels(window)) if mip != 0)
        assert others

        def each(step, act):
            resident = _mip0_chunks(window)
            for spy in (rect_patches, evictions, regions):
                del spy[:]
            act()
            visible = [bbox for bbox, levels in evictions if levels is None or 0 in levels]
            assert [levels for _bbox, levels in evictions if levels is not None and 0 not in levels] == [others], (
                f"{step}: {evictions}"
            )
            if style == "Sloped":
                assert rect_patches == [] and visible == [regions[-1]], f"{step}: {rect_patches} {visible}"
            else:
                assert visible == [], f"{step}: evicted {visible} at the visible level"
                (call,) = rect_patches
                assert call["levels"] == (0,) and call["terrain_current"], f"{step}: {call}"
                rects, bbox = call["rects"], regions[-1]
                assert len(rects) > 1, f"{step}: {rects}"
                assert all(_inside(bbox, rect) for rect in rects), f"{step}: a rect leaves the repainted {bbox}"
                area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)
                assert area < (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), f"{step}: the rects save nothing"
                # Every chunk is resident, so each per-chunk rect is exactly one composite.
                assert len(call["composites"]) == len(rects), f"{step}: {call['composites']}"
                assert _mip0_chunks(window) == resident, f"{step}: the visible level's chunks changed"
            _check_against_fresh(window, step)

        _long_batches(window, each)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_rects_repaint_the_same_qt_region_as_the_eviction(blended_trees, monkeypatch) -> None:
    """What Qt is told to repaint does not depend on how the visible level's
    chunks were brought up to date: the same strokes, undo and redo ask for
    the same bbox whether the batch patched rects or evicted."""
    _seed_planner(monkeypatch)
    asked = {}
    for repaint in ("rects", "evict"):
        window = _blended_window("Stepped", monkeypatch, repaint)
        try:
            regions = _spy_qt_regions(window, monkeypatch)
            rect_patches = _spy_rect_patches(window, monkeypatch)
            asked[repaint] = []

            def each(step, act, regions=regions, repaint=repaint):
                act()
                asked[repaint].append((step, regions[-1]))

            _long_batches(window, each)
            assert len(rect_patches) == (len(LONG_STEPS) if repaint == "rects" else 0)
        finally:
            window.edit_history.mark_saved()
            window.close()
    assert asked["rects"] == asked["evict"]


@pytest.mark.parametrize("broken", BLENDED_STEPS)
def test_the_blended_check_names_the_one_step_whose_rects_fell_short(broken, blended_trees, monkeypatch) -> None:
    """Control (GOTCHAS: a step-k break fails naming step k): on step `broken`
    only, every per-chunk rect loses its right half, where the stroke's oaks
    stand on what was grass (on the left they replace palms, and the
    synthetic art gives every tree the same sprite)."""
    window = _blended_window("Stepped", monkeypatch, "rects")
    try:
        real = window._unit_edit_chunk_rects
        current = {}

        def chunk_rects(*args, **kwargs):
            rects = real(*args, **kwargs)
            if current.get("step") != broken:
                return rects
            return [(x0, y0, (x0 + x1) // 2, y1) for x0, y0, x1, y1 in rects]

        monkeypatch.setattr(window, "_unit_edit_chunk_rects", chunk_rects)
        with pytest.raises(AssertionError, match=f"{broken}: mip 0 differs"):
            _blended_history(window, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_one_tile_wide_stroke_edge_is_repainted_with_its_blended_neighbours(blended_trees, monkeypatch) -> None:
    """Undo and redo of a stroke whose terrain includes a one-tile-wide run
    with no unit on it, in a block of tiles of its own: under blending its 8
    neighbours change with it, and only the one-tile dilation inside
    dirty_screen_bbox_iso() puts them in that block's rect."""
    from descape import render

    window = _blended_window("Stepped", monkeypatch, "rects")
    try:
        assert render.blend_effective(window._cache.layers)
        rect_patches = _spy_rect_patches(window, monkeypatch)
        _blocked_stroke_history(window, monkeypatch, block=EDGE_TILES)
        # The undo and the redo (the stroke itself, with no chunk resident yet, takes the reach fallback).
        assert len(rect_patches) == 2 and all(call["composites"] for call in rect_patches), rect_patches
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", ["undo", "redo"])
def test_the_stroke_edge_check_fails_if_a_block_rect_loses_its_one_tile_ring(
    broken, blended_trees, monkeypatch
) -> None:
    """Control: on step `broken` only, each block's rect is sized from its
    dirty tiles as they are, without the one-tile dilation
    dirty_screen_bboxes_iso_grouped() gives them, as dirty_screen_bbox_iso()
    does (the same sweep, over the dirty tiles alone). The edge run's blended
    neighbours then stay stale."""
    import descape.viewer as viewer_module
    from descape import iso_geometry, render

    window = _blended_window("Stepped", monkeypatch, "rects")
    try:
        assert render.blend_effective(window._cache.layers)
        real_grouped = viewer_module.dirty_screen_bboxes_iso_grouped
        state = {"bare": 0}
        current = {}

        def bare_grouped(scenario, groups, elevations, proj, **kwargs):
            boxes = real_grouped(scenario, groups, elevations, proj, **kwargs)
            if current.get("step") != broken:
                return boxes
            width = scenario.map_manager.map_width
            bare = []
            for group, box in zip(groups, boxes, strict=True):
                idx = np.unique(np.asarray(list(group), dtype=np.int64))
                xs, ys = idx % width, idx // width
                lo, hi = render._observed_elevation_range(elevations, {}, xs, ys)
                x0, y0, x1, y1 = iso_geometry.tile_screen_bounds_over(xs, ys, lo, hi, proj)
                bare.append((int(x0.min()), int(y0.min()), int(x1.max()), int(y1.max())))
                assert _inside(box, bare[-1]) and bare[-1] != box, (box, bare[-1])
            state["bare"] += len(bare)
            return bare

        monkeypatch.setattr(viewer_module, "dirty_screen_bboxes_iso_grouped", bare_grouped)
        with pytest.raises(AssertionError, match=f"{broken}: mip 0 differs"):
            _blocked_stroke_history(
                window, monkeypatch, before_step=lambda step: current.update(step=step), block=EDGE_TILES
            )
        # One block for the edge run, one for the trees.
        assert state["bare"] == 2, state
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("history", ["long", "blocked"])
@pytest.mark.parametrize("style", ["Stepped", "FlatIso"])
def test_the_block_rects_are_what_one_bbox_call_per_block_sizes(style, history, blended_trees, monkeypatch) -> None:
    """The grouped pass and the per-block dirty_screen_bbox_iso() calls it
    replaced (still the path when it declines) hand cache.chunk_rects() the
    same rects: over two long batches, an undo and a redo (`long`), and over
    a history move that carries terrain tiles past its units (`blocked`)."""
    import descape.viewer as viewer_module

    window = _blended_window(style, monkeypatch, "rects")
    try:
        real_rects, real_chunk = window._unit_edit_chunk_rects, window._cache.chunk_rects
        real_bbox = viewer_module.dirty_screen_bbox_iso
        seen = {"sized": [], "per_block": 0, "compared": 0}

        def chunk_rects(rects, mip):
            seen["sized"].append(sorted(rects))
            return real_chunk(rects, mip)

        def per_block(*args, **kwargs):
            seen["per_block"] += 1
            return real_bbox(*args, **kwargs)

        def both(blocks, terrain_tiles, visible_mip):
            del seen["sized"][:]
            seen["per_block"] = 0
            with monkeypatch.context() as patch:
                patch.setattr(viewer_module, "dirty_screen_bboxes_iso_grouped", lambda *a, **k: None)
                patch.setattr(viewer_module, "dirty_screen_bbox_iso", per_block)
                slow = real_rects(blocks, terrain_tiles, visible_mip)
            assert seen["per_block"] >= 1, "the declined pass made no per-block call"
            calls = seen["per_block"]
            fast = real_rects(blocks, terrain_tiles, visible_mip)
            assert seen["per_block"] == calls, "the grouped pass fell back to the per-block calls"
            assert seen["sized"][0] == seen["sized"][1] and seen["sized"][0], seen["sized"]
            assert fast == slow and fast
            seen["compared"] += 1
            return fast

        monkeypatch.setattr(window._cache, "chunk_rects", chunk_rects)
        monkeypatch.setattr(window, "_unit_edit_chunk_rects", both)
        if history == "long":
            _long_batches(window, lambda step, act: (act(), _check_against_fresh(window, step)))
            assert seen["compared"] == len(LONG_STEPS)
        else:
            _blocked_stroke_history(window, monkeypatch, block=EDGE_TILES)
            # The undo and the redo (the stroke itself, with no chunk resident yet, takes the reach fallback).
            assert seen["compared"] == 2, seen["compared"]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("side", ["within", "past"])
def test_a_batch_patches_its_rects_up_to_the_rect_ratio_and_evicts_past_it(side, blended_trees, monkeypatch) -> None:
    """The production choice at a finite ratio: for each of LONG_STEPS'
    batches the ratio is set half a level pixel either side of what its own
    rects price at. Within it the batch patches them; past it, it evicts the
    union bbox and patches nothing. Every level matches a fresh build either
    way."""
    import descape.viewer as viewer_module

    window = _blended_window("Stepped", monkeypatch, "rects")
    try:
        cache = window._cache
        rect_patches = _spy_rect_patches(window, monkeypatch)
        evictions = _spy_evictions(window, monkeypatch)
        real = window._unit_edit_chunk_rects
        priced = []

        def chunk_rects(blocks, terrain_tiles, visible_mip):
            rects = real(blocks, terrain_tiles, visible_mip)
            _mip, cx0, cy0, cx1, cy1 = window.map_view.viewport_chunk_target()
            visible = (cx1 - cx0 + 1) * (cy1 - cy0 + 1) * cache.chunk_px**2
            area = sum(cache.patch_area(rect, levels=(visible_mip,)) for rect in rects)
            assert 0 < area < visible, (area, visible)
            priced.append(area / visible)
            nudge = 0.5 if side == "within" else -0.5
            monkeypatch.setattr(viewer_module, "_RECT_PATCH_AREA_RATIO", (area + nudge) / visible)
            return rects

        monkeypatch.setattr(window, "_unit_edit_chunk_rects", chunk_rects)

        def each(step, act):
            del rect_patches[:], evictions[:]
            sized = len(priced)
            act()
            assert len(priced) == sized + 1, f"{step}: the batch never sized its rects"
            if side == "within":
                assert len(rect_patches) == 1 and evictions == [], f"{step}: {rect_patches} {evictions}"
            else:
                assert rect_patches == [] and len(evictions) == 1, f"{step}: {rect_patches} {evictions}"
            _check_against_fresh(window, step)

        _long_batches(window, each)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", ["undo", "redo"])
def test_the_blocked_stroke_check_names_the_step_whose_terrain_tiles_were_dropped(
    broken, tree_sprites, monkeypatch
) -> None:
    """Control for the rects' terrain term: on step `broken` only, the split
    gets the history move's terrain bbox but not its tiles, so the rects are
    sized from the units alone and the terrain past them stays stale."""
    _pin_area_ratio(monkeypatch, 0, rects=float("inf"))
    window = _over_cap_window("Stepped", monkeypatch)
    try:
        real = window._repaint_unit_edit_split
        current = {}

        def split(*args, terrain_tiles=None, **kwargs):
            return real(*args, terrain_tiles=None if current.get("step") == broken else terrain_tiles, **kwargs)

        monkeypatch.setattr(window, "_repaint_unit_edit_split", split)
        with pytest.raises(AssertionError, match=f"{broken}: mip 0 differs"):
            _blocked_stroke_history(window, monkeypatch, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_terrain_bbox_that_comes_without_its_tiles_keeps_the_eviction(tree_sprites, monkeypatch) -> None:
    """A caller that defers a terrain bbox but hands over no tiles cannot be
    sized per block, so the batch evicts the union bbox as before."""
    from descape.render_cache import UnitSplice

    _pin_area_ratio(monkeypatch, 0, rects=float("inf"))
    window = _over_cap_window("Stepped", monkeypatch)
    try:
        _render_levels(window)
        rect_patches = _spy_rect_patches(window, monkeypatch)
        evictions = _spy_evictions(window, monkeypatch)
        mm = window.scenario.map_manager
        dirty = [y * mm.map_width + x for x, y in STROKE_TILES]
        for i in dirty:
            mm.terrain[i].terrain_id = FOREST_PALM
        deferred: list = []
        assert window._apply_dirty(dirty, deferred=deferred) is False and len(deferred) == 1
        model = window._ensure_unit_edits()
        model.begin_unit_edit([1])
        unit = model.add(1, CONST, 60.5, 60.5)
        model.commit_unit_edit("Add", window.edit_history)
        own, tiles = window._unit_footprint(unit)
        window._after_unit_mutation(
            [UnitSplice(1, 0, unit, None, own, (), tiles)], batch=True, terrain_bbox=deferred[0]
        )
        assert rect_patches == [] and len(evictions) == 1, (rect_patches, evictions)
        _check_against_fresh(window, "no tiles")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_over_ratio_erase_flush_re_reads_the_layer_under_its_rects(blended_trees, monkeypatch) -> None:
    """The Erase flush and its stroke end vouch for no terrain (Terrain
    Overlay wrote `layer = -1` live), so their rects re-read both mirrors
    before compositing, the visible level being the only resident one, so
    with no other eviction to re-read them first. Pinned by the flag alone:
    each call's own _apply_dirty() over the cleared tiles shows the layer
    whether or not the rects re-read."""
    import sys

    from descape import render

    # One oracle level: _render_levels() and _check_against_fresh() then leave no second level resident.
    monkeypatch.setattr(sys.modules[__name__], "_oracle_levels", lambda _window: (0,))
    window = _blended_window("Stepped", monkeypatch, "rects")
    try:
        assert render.blend_effective(window._cache.layers)
        rect_patches = _spy_rect_patches(window, monkeypatch)
        evictions = _spy_all_evictions(window, monkeypatch)
        _overlay_erase(window)
        assert window._cache.resident_levels() == [0]
        assert evictions == [], evictions
        # The flush and the Erase stroke's end (the setup stroke, with no chunk resident yet, took the reach fallback).
        assert [call["terrain_current"] for call in rect_patches] == [False, False], rect_patches
        assert all(call["composites"] for call in rect_patches)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rects_skip_a_chunk_that_is_not_resident_and_patch_resident_ones_off_the_viewport(
    blended_trees, monkeypatch
) -> None:
    """The viewport shows one chunk of a fully resident level, except one
    chunk under the stroke, which is not cached. The rects are composited
    into every resident chunk they reach, on screen or not, and never into
    the missing one, which stays missing until it is next asked for."""
    from descape import iso_geometry

    window = _blended_window("Stepped", monkeypatch, "rects")
    try:
        cache = window._cache
        _draw(window, FOREST_OAK, LONG_TILES)
        _render_levels(window)
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (0, 0, 0, 0, 0))
        size = cache.chunk_px
        px, py = (int(v) for v in iso_geometry.tile_screen_origin(*LONG_SHIFTED[30], 0, cache.proj))
        missing = (0, px // size, py // size)
        cache.invalidate_region((px, py, px + 1, py + 1), levels=(0,), terrain_changed=False)
        assert missing not in cache._cache and missing[1:] != (0, 0)
        resident = _mip0_chunks(window)
        rect_patches = _spy_rect_patches(window, monkeypatch)
        evictions = _spy_evictions(window, monkeypatch)

        _draw(window, FOREST_PALM, LONG_SHIFTED)

        (call,) = rect_patches
        assert evictions == []
        ox, oy = missing[1] * size, missing[2] * size
        under = [r for r in call["rects"] if r[0] < ox + size and r[2] > ox and r[1] < oy + size and r[3] > oy]
        assert under, "no rect reached the missing chunk, so nothing was at risk"
        assert _mip0_chunks(window) == resident, "a chunk was evicted, or the missing one was built"
        patched = {(x0 // size, y0 // size) for mip, x0, y0, _x1, _y1 in call["composites"] if mip == 0}
        assert missing[1:] not in patched
        assert patched - {(0, 0)}, f"no resident chunk off the viewport was patched: {patched}"
        _check_against_fresh(window, "stroke")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_chunk_rects_keeps_one_rect_per_chunk_and_clips_to_the_level(tree_sprites, monkeypatch) -> None:
    """cache.chunk_rects(): overlapping rects in one chunk become their bbox,
    a rect across a chunk edge one rect on each side, and whatever lies off
    the level's canvas is dropped. At a coarser level the answer is still in
    reference pixels."""
    window = _over_cap_window("Stepped", monkeypatch)
    try:
        cache = window._cache
        size = cache.chunk_px
        canvas_w, canvas_h = cache.canvas_dims(0)
        assert canvas_w > 2 * size and canvas_h > size
        rects = [(10, 10, 60, 40), (50, 30, 90, 80), (size - 20, 100, size + 30, 140)]
        assert cache.chunk_rects(rects, 0) == [
            (10, 10, size, 140), (size, 100, size + 30, 140)
        ]
        assert cache.chunk_rects([(-50, -50, 20, 30), (canvas_w - 5, 0, canvas_w + 99, 9)], 0) == [
            (0, 0, 20, 30), (canvas_w - 5, 0, canvas_w, 9)
        ]
        assert cache.chunk_rects([(-9, -9, 0, 0), (canvas_w, 0, canvas_w + 5, 5), (7, 7, 7, 9)], 0) == []
        scale = cache.mip_tile_px(0) // cache.mip_tile_px(-1)
        assert scale > 1
        edge = size * scale
        assert cache.chunk_rects([(edge - scale, 0, edge + scale, scale)], -1) == [
            (edge - scale, 0, edge, scale), (edge, 0, edge + scale, scale)
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- Past both caps the visible level splices alone (TASK-031.61 c2x) ---

WIDE = 10**6  # VISIBLE_SPLICE_MAX_UNITS live, far past every batch here


def _spy_visible_splices(window, monkeypatch) -> list:
    """Each cache._splice_visible() answer from here on, in order."""
    answers = []
    real = window._cache._splice_visible
    monkeypatch.setattr(window._cache, "_splice_visible", lambda *a: (answers.append(real(*a)), answers[-1])[1])
    return answers


def _count_builds_and_walks(monkeypatch) -> dict:
    """Calls into the two sliced walks every level build runs with sprites on
    (`builds`), and into render._units_by_tile() (`walks`)."""
    from descape import render

    counts = {"builds": 0, "walks": 0}

    def counted(name: str, key: str) -> None:
        real = getattr(render, name)

        def call(*args, **kwargs):
            counts[key] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(render, name, call)

    counted("_building_bboxes_iso_sliced", "builds")
    counted("sprite_draws_by_anchor_sliced", "builds")
    counted("_units_by_tile", "walks")
    return counts


def _spy_phase_names(monkeypatch) -> list:
    """Every Perf Trace phase name opened from here on, trace on or off."""
    from descape import perf_trace

    names = []
    real = perf_trace.phase
    monkeypatch.setattr(perf_trace, "phase", lambda name, *a, **k: (names.append(name), real(name, *a, **k))[1])
    return names


def _wide_history(window, monkeypatch, before_step=None) -> None:
    """_blended_history()'s steps (a stroke of 8 oaks over 4 of 8 palms, its
    undo, which puts palms back mid-list, and its redo), each past OVER_CAP.
    Each step and the next paint of the visible level must build no level and
    walk no unit, open the splice_visible= phase, and leave every oracle
    level equal to a fresh build, failing as `<step>: ...`."""
    _draw(window, FOREST_PALM, EARLIER_TILES)
    acts = {"stroke": lambda: _draw(window, FOREST_OAK, STROKE_TILES), "undo": window.undo, "redo": window.redo}
    counts = _count_builds_and_walks(monkeypatch)
    phases = _spy_phase_names(monkeypatch)
    cache = window._cache
    for step in BLENDED_STEPS:
        _render_levels(window)
        if before_step is not None:
            before_step(step)
        counts.update(builds=0, walks=0)
        del phases[:]
        acts[step]()
        cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)
        assert counts == {"builds": 0, "walks": 0}, f"{step}: {counts}"
        tokens = [n.split("=")[0] for n in phases if n.startswith("splice_")]
        assert tokens == ["splice_visible"], f"{step}: {phases}"
        _check_against_fresh(window, step)


@pytest.mark.parametrize("repaint", ["patch", "evict", "rects"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_past_the_caps_a_draw_its_undo_and_redo_splice_the_visible_level_and_match_a_fresh_render(
    style, repaint, blended_trees, monkeypatch
) -> None:
    """The wide bound live, sprites and blending on: the stroke end, its undo
    and its redo each splice the visible level, which the patch, the
    eviction or Stepped's per-chunk rects then composite as current."""
    window = _blended_window(style, monkeypatch, repaint, visible=WIDE)
    try:
        _wide_history(window, monkeypatch)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", BLENDED_STEPS)
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_wide_splice_check_names_the_one_step_whose_visible_level_was_not_spliced(
    style, broken, blended_trees, monkeypatch
) -> None:
    """Control (GOTCHAS: a step-k break fails naming step k): on step
    `broken` only, the visible level is restamped current but not spliced."""
    window = _blended_window(style, monkeypatch, visible=WIDE)
    try:
        cache = window._cache
        real = cache._splice_level
        current = {}
        monkeypatch.setattr(
            cache, "_splice_level", lambda *a: None if current.get("step") == broken else real(*a)
        )
        with pytest.raises(AssertionError, match=f"{broken}: mip 0 differs"):
            _wide_history(window, monkeypatch, before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_past_the_caps_a_sprites_off_stroke_end_keeps_the_in_place_update(style, tree_sprites, monkeypatch) -> None:
    """With sprites off a level build is cheap, so the wide splice declines
    and both strokes take today's in-place update, matching a fresh build."""
    window = _over_cap_window(style, monkeypatch, visible=WIDE)
    try:
        window.show_sprites_action.setChecked(False)
        assert not window._cache.sprites_enabled
        calls = _spy(window, monkeypatch)
        answers = _spy_visible_splices(window, monkeypatch)
        _over_cap_strokes(window)
        assert answers == [False, False] and calls["in_place"] == 2 and calls["wholesale"] == 0, (answers, calls)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_past_the_caps_a_terrain_overlay_erase_flush_matches_a_fresh_render(style, blended_trees, monkeypatch) -> None:
    """An Eraser flush of 6 oaks (past OVER_CAP), two of them on layered
    tiles, Trees and Terrain Overlay ticked: it splices the visible level and
    still repaints both cleared layers."""
    window = _blended_window(style, monkeypatch, visible=WIDE)
    try:
        _draw(window, FOREST_OAK, STROKE_TILES)
        mm = window.scenario.map_manager
        layered = [y * mm.map_width + x for x, y in (ERASED_FIRST, ERASED_LAST)]
        for i in layered:
            mm.terrain[i].layer = LAYER_ID
        window._apply_dirty(layered)
        _render_levels(window)
        _check_against_fresh(window, "layered")
        trees = len(window.scenario.unit_manager.units[0])
        answers = _spy_visible_splices(window, monkeypatch)
        _open_erase_stroke(window, trees=True, overlay=True)
        assert window._erase.overlay, "Terrain Overlay does not apply to this file"
        window.on_edit_stroke_tiles(STROKE_TILES[:6], 0)
        window._erase_refresh_timer.stop()
        window._flush_erase_refresh()
        assert window._erase is not None  # still mid-stroke
        assert all(mm.terrain[i].layer == -1 for i in layered)
        assert len(window.scenario.unit_manager.units[0]) == trees - 6
        assert answers == [True], answers
        _check_against_fresh(window, "erase flush")
        window.on_edit_stroke_end()
        _check_against_fresh(window, "erase end")
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_past_the_caps_a_second_undo_before_any_paint_matches_a_fresh_render(style, tree_sprites, monkeypatch) -> None:
    """Two over-cap undos with no paint between them both splice the visible
    level, the second over the first's spliced layer; so do the two redos."""
    window = _over_cap_window(style, monkeypatch, visible=WIDE)
    try:
        _draw(window, FOREST_OAK, STROKE_TILES)
        _draw(window, FOREST_PALM, EARLIER_TILES)
        _render_levels(window)
        answers = _spy_visible_splices(window, monkeypatch)
        window.undo()
        window.undo()
        assert answers == [True, True], answers
        _check_against_fresh(window, "two undos")
        window.redo()
        window.redo()
        assert answers == [True] * 4, answers
        _check_against_fresh(window, "two redos")
    finally:
        window.edit_history.mark_saved()
        window.close()
