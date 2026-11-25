"""A one-shot tool edit that raises partway rolls back and repaints (TASK-503).

Each test arms one model method (a call counter, never monkeypatch.undo()),
re-warms the whole mip-0 canvas, runs the gesture under pytest.raises, then
checks in this order: the document equals its pre-gesture snapshot (every
tile's state, each player's unit list by identity plus unit_state()), the
history is unchanged, no unit edit is left open, and the canvas equals a
fresh render. One more ordinary unit edit then succeeds. Stepped, on the
blank template; Paste Region's own cases live in test_region_paste_render.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import render_iso, unit_pick, unit_rotation
from descape.edit_history import tile_state
from descape.unit_model import unit_state
from testkit.viewer_family import set_viewer_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

VILLAGER = 83
FOREST_OAK = 10  # density 1000/1000: one tree per tile
FOREST_PALM = 13  # another forest, so a stroke over FOREST_OAK removes and adds
FOREST_OAK_CONST = 411
STROKE_TILES = [(20 + i, 30) for i in range(8)]
SHIFTED_TILES = [(24 + i, 30) for i in range(8)]


def arm(monkeypatch, obj, name: str, on_call: int = 1, through: bool = False) -> list[int]:
    """`obj.name` calls through, except its `on_call`th call raises RuntimeError("armed"),
    after calling through first when `through`. Returns the call counter."""
    real = getattr(obj, name)
    calls = [0]

    def armed(*args, **kwargs):
        calls[0] += 1
        if calls[0] != on_call:
            return real(*args, **kwargs)
        if through:
            real(*args, **kwargs)
        raise RuntimeError("armed")

    monkeypatch.setattr(obj, name, armed)
    return calls


def canvas(window) -> np.ndarray:
    canvas_w, canvas_h = window._cache.canvas_dims(0)
    return window._cache.render_rect(0, 0, canvas_w, canvas_h, mip=0).copy()


def fresh(window) -> np.ndarray:
    canvas_w, canvas_h = window._cache.canvas_dims(0)
    full = render_iso.render_terrain_iso_with_proj(
        window.scenario, with_units=True, with_sprites=window._cache.sprites_enabled
    )[0]
    return full[:canvas_h, :canvas_w]


def snapshot(window) -> tuple:
    """The document (tiles, unit lists, unit states) and the history (cursor, records)."""
    lists = [list(units) for units in window.scenario.unit_manager.units]
    history = window.edit_history
    return (
        [tile_state(t) for t in window.scenario.map_manager.terrain],
        lists,
        [[unit_state(u) for u in units] for units in lists],
        history.cursor,
        list(history.records),
    )


def assert_rolled_back(window, then) -> None:
    tiles, lists, states, cursor, records = then
    now = window.scenario.unit_manager.units
    assert [tile_state(t) for t in window.scenario.map_manager.terrain] == tiles, "tiles not put back"
    assert [len(units) for units in now] == [len(units) for units in lists], "unit lists not put back"
    assert all(a is b for old, new in zip(lists, now, strict=True) for a, b in zip(old, new, strict=True))
    assert [[unit_state(u) for u in units] for units in now] == states, "unit states not put back"
    history = window.edit_history
    assert history.cursor == cursor
    assert len(history.records) == len(records)
    assert all(a is b for a, b in zip(history.records, records, strict=True))
    assert window.unit_edits._pending is None, "the unit edit was left open"
    assert np.array_equal(canvas(window), fresh(window)), "the canvas does not match a fresh render"


def assert_next_edit_works(window) -> None:
    model = window._ensure_unit_edits()
    records = len(window.edit_history.records)
    with window._unit_edit(model, "Add", [1]):
        model.add(1, VILLAGER, 50.5, 50.5)
    assert len(window.edit_history.records) == records + 1


def _window():
    window = conftest.terrain_edit_window()
    window.terrain_style_combo.setCurrentText("Stepped")
    return window


def _seed_villagers(window, count: int) -> list:
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "seed", [1]):
        return [model.add(1, VILLAGER, 30.5 + 2 * i, 30.5) for i in range(count)]


def test_a_rotate_that_raises_on_its_second_unit_rolls_back_and_repaints(monkeypatch) -> None:
    """_unit_edit()'s fields_only path: the first unit's facing is put back."""
    assert unit_rotation.rotation_is_angle(VILLAGER)
    window = _window()
    try:
        units = _seed_villagers(window, 2)
        window.mode_combo.setCurrentText("Units")
        window._selection = [unit_pick.unit_key(1, u) for u in units]
        window._refresh_selection_view()
        arm(monkeypatch, window.unit_edits, "set_rotation", on_call=2)
        then = snapshot(window)
        canvas(window)
        with pytest.raises(RuntimeError, match="armed"):
            window.on_unit_rotate(1)
        assert_rolled_back(window, then)
        assert_next_edit_works(window)
    finally:
        conftest.close_window(window)


def test_a_scatter_that_raises_after_its_adds_rolls_back_and_repaints(monkeypatch) -> None:
    """_unit_edit()'s whole-list path: Scatter Units in Region's placed units are taken back out."""
    from PyQt5.QtWidgets import QDialog

    from descape.scatter_dialog import ScatterDialog

    window = _window()
    try:
        window.units_panel.select_owner(1)
        window.units_panel.select_object(VILLAGER)
        window.on_region_selected((60, 60, 70, 70))

        def accept(dialog):
            dialog.count_spin.setValue(5)
            return QDialog.Accepted

        monkeypatch.setattr(ScatterDialog, "exec_", accept)
        calls = arm(monkeypatch, window._ensure_unit_edits(), "add_many", through=True)
        then = snapshot(window)
        canvas(window)
        with pytest.raises(RuntimeError, match="armed"):
            window.scatter_units_in_region()
        assert calls == [1], "the scatter never reached add_many()"
        assert_rolled_back(window, then)
        assert_next_edit_works(window)
    finally:
        conftest.close_window(window)


def _paint_oaks(window) -> None:
    window._on_tool_selected("draw")
    window.paint_trees_check.setChecked(True)
    window.paint_eye_candy_check.setChecked(False)
    window.terrain_panel.set_terrain(FOREST_OAK)
    window.on_edit_stroke_start()
    for x, y in STROKE_TILES:
        window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()
    assert sum(u.unit_const == FOREST_OAK_CONST for u in window.scenario.unit_manager.units[0]) == len(STROKE_TILES)


def test_a_draw_stroke_whose_trees_raise_rolls_back_the_stroke_and_repaints(monkeypatch) -> None:
    """_apply_terrain_unit_plan(): the stroke painted live, removed oaks and
    added palms, then raised; tiles and trees both go back."""
    window = _window()
    try:
        _paint_oaks(window)
        calls = arm(monkeypatch, window.unit_edits, "add_many", through=True)
        then = snapshot(window)
        canvas(window)
        window.terrain_panel.set_terrain(FOREST_PALM)
        window.on_edit_stroke_start()
        for x, y in SHIFTED_TILES:
            window.on_edit_stroke_tile(x, y, 0)
        with pytest.raises(RuntimeError, match="armed"):
            window.on_edit_stroke_end()
        assert calls == [1]
        assert_rolled_back(window, then)
        assert_next_edit_works(window)
    finally:
        conftest.close_window(window)


def test_a_paint_can_fill_whose_trees_raise_rolls_back_the_fill_and_repaints(monkeypatch) -> None:
    set_viewer_global(monkeypatch, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 100_000)
    window = _window()
    try:
        _paint_oaks(window)
        window._on_tool_selected("fill")
        window.terrain_panel.set_terrain(FOREST_PALM)
        calls = arm(monkeypatch, window.unit_edits, "add_many", through=True)
        then = snapshot(window)
        canvas(window)
        with pytest.raises(RuntimeError, match="armed"):
            window.on_fill(*STROKE_TILES[0], 0)
        assert calls == [1]
        assert_rolled_back(window, then)
        assert not window._busy and window.isEnabled()
        assert_next_edit_works(window)
    finally:
        conftest.close_window(window)


def test_a_mirror_map_whose_second_image_raises_rolls_back_and_repaints(monkeypatch) -> None:
    """_apply_mirror_units(): the removed destination tree and the first image are both put back."""
    from descape.mirror_tools import plan_mirror, plan_mirror_units

    window = _window()
    try:
        mm = window.scenario.map_manager
        w = mm.map_width
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [0]):
            model.add(0, FOREST_OAK_CONST, 10.5, 12.5)
            model.add(0, FOREST_OAK_CONST, w - 20.5, w - 30.5)
            model.add(0, FOREST_OAK_CONST, 14.5, 20.5)
        plan = plan_mirror(mm, 1, 0, do_terrain=True, do_elevation=False)
        unit_plan = plan_mirror_units(
            mm, 1, 0, window.scenario.unit_manager.units, plan.source_indices,
            referencing=window.unit_edits.referencing,
        )
        assert unit_plan.removals and len(unit_plan.images) >= 2
        calls = arm(monkeypatch, model, "add", on_call=2)
        then = snapshot(window)
        canvas(window)
        with pytest.raises(RuntimeError, match="armed"):
            window.on_mirror(plan, unit_plan)
        assert calls == [2]
        assert_rolled_back(window, then)
        assert not window._busy and window.isEnabled()
        assert_next_edit_works(window)
    finally:
        conftest.close_window(window)
