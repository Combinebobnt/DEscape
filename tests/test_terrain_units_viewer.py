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

from descape import settings
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
    """Records every _after_unit_mutation `changed` argument and every
    wholesale source rebuild: a _refresh_source_caches() call from inside
    invalidate_units(), which every style's splice path never makes (patch()
    and Flat's patch_rects() call it too, so only calls inside count)."""
    calls = {"changed": [], "wholesale": 0}
    inside = [False]
    real_after = window._after_unit_mutation
    cache = window._cache
    real_invalidate, real_refresh = cache.invalidate_units, cache._refresh_source_caches

    def after(changed=None, **kwargs):
        calls["changed"].append(changed)
        return real_after(changed, **kwargs)

    def invalidate(changed=None):
        inside[0] = True
        try:
            return real_invalidate(changed)
        finally:
            inside[0] = False

    def refresh(elevation_changed=None):
        calls["wholesale"] += inside[0]
        return real_refresh(elevation_changed)

    monkeypatch.setattr(window, "_after_unit_mutation", after)
    monkeypatch.setattr(cache, "invalidate_units", invalidate)
    monkeypatch.setattr(cache, "_refresh_source_caches", refresh)
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
    """Every bbox window._cache.invalidate_region() evicts from here on."""
    evictions = []
    real_evict = window._cache.invalidate_region

    def evict(bbox, levels=None):
        evictions.append(bbox)
        return real_evict(bbox, levels=levels)

    monkeypatch.setattr(window._cache, "invalidate_region", evict)
    return evictions


@pytest.mark.parametrize("repaint", ["patch", "evict"])
@pytest.mark.parametrize("sprites", [False, True], ids=["marks", "sprites"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_draw_over_a_tile_another_unit_holds_repaints_only_its_bbox(style, sprites, repaint, monkeypatch, request) -> None:
    """A contested tile makes the cache refuse the splice, so invalidate_units()
    rebuilds its sources (from the sprite memo), and the stroke end still
    repaints only the batch's bbox, eagerly or by evicting its chunks: never
    the whole canvas. Undo and redo take the same path. With sprites, the
    contested unit is a real sprite, so the memo-backed rebuild is what the
    pixel oracle checks."""
    if sprites:
        request.getfixturevalue("sprite_install")
    _splice_any_batch(monkeypatch)
    if repaint == "evict":
        _pin_area_ratio(monkeypatch, 0)
    window = _styled_window(style)
    try:
        # A configured real install turns sprites on by default, so set both states explicitly.
        window.show_sprites_action.setChecked(sprites)
        assert window._cache.sprites_enabled is sprites
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
        assert calls["wholesale"] == 1, "expected the contested batch to rebuild its sources"
        assert bool(evictions) is (repaint == "evict")
        assert canvas not in evictions
        after = _canvas(window)
        assert not np.array_equal(after, before)
        assert np.array_equal(after, _fresh(window, style))
        window.undo()
        assert calls["changed"][-1]
        assert np.array_equal(_canvas(window), before)
        assert np.array_equal(_canvas(window), _fresh(window, style))
        window.redo()
        assert calls["changed"][-1]
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


def test_undo_of_a_non_gaia_edit_stays_wholesale(monkeypatch) -> None:
    """The membership diff only covers GAIA-only records."""
    window = _styled_window("Stepped")
    try:
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [1]):
            model.add(1, 83, 40.5, 40.5)
        calls = _spy(window, monkeypatch)
        window.undo()
        assert calls["changed"] == [None]
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
