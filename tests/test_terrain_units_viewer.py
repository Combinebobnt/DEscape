"""descape/terrain_units.py's ViewerWindow wiring: the Trees/Eye candy
checkboxes, the composite-undo path Draw and Paint Can both feed into, the
Paint Can large-fill confirm guard, and settings persistence. Same offscreen
technique tests/test_fill_tool.py documents; every ViewerWindow() here must
call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sprite_toggle_viewer import sprite_install  # noqa: F401 -- fixture, requested by name

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
    window = _window("draw")
    if style == "Flat":
        # Top-down Flat, not the Flat+Isometric path (see test_unit_edit_viewer's _window).
        window.iso_action.setChecked(False)
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

    def invalidate(changed=None, splice_levels=True):
        inside[0] = True
        try:
            return real_invalidate(changed, splice_levels=splice_levels)
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
    than patch (viewer._SCOPED_/_TIGHT_PATCH_AREA_RATIO), so pin both off."""
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "_SPLICE_COST_RATIO", 0)
    _pin_area_ratio(monkeypatch, float("inf"))


def _pin_area_ratio(monkeypatch, value: float) -> None:
    """Sets both the reach-path and the tight-split area ratio, for every style."""
    import descape.viewer as viewer_module

    for name in ("_SCOPED_PATCH_AREA_RATIO", "_TIGHT_PATCH_AREA_RATIO"):
        monkeypatch.setattr(viewer_module, name, {"stepped": value, "sloped": value})


@pytest.mark.parametrize("style", ["Stepped", "Sloped", "Flat"])
def test_draw_through_existing_trees_splices_and_repaints_like_a_fresh_render(style, monkeypatch) -> None:
    _splice_any_batch(monkeypatch)
    window = _styled_window(style)
    try:
        _draw(window, FOREST_OAK)
        before = _canvas(window)  # every chunk resident, so a stale one would show
        assert np.array_equal(before, _fresh(window, style))
        calls = _spy(window, monkeypatch)

        _draw(window, FOREST_PALM, SHIFTED_TILES)

        (changed,) = calls["changed"]
        assert changed, "the stroke end took the wholesale unit invalidation"
        removed = {s.old_own_tile for s in changed if s.new_own_tile is None}
        assert removed and removed <= {s.new_own_tile for s in changed}, "expected remove + add on shared tiles"
        assert calls["wholesale"] == 0, "the cache fell back to a wholesale source rebuild"
        after = _canvas(window)
        assert not np.array_equal(after, before), "the new trees changed nothing on screen"
        assert np.array_equal(after, _fresh(window, style))

        window.undo()
        assert calls["changed"][-1], "undo took the wholesale unit invalidation"
        assert np.array_equal(_canvas(window), _fresh(window, style))
        assert np.array_equal(_canvas(window), before)
        window.redo()
        assert calls["changed"][-1], "redo took the wholesale unit invalidation"
        assert calls["wholesale"] == 0
        assert np.array_equal(_canvas(window), after)
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
