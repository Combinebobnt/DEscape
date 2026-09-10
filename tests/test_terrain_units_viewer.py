"""descape/terrain_units.py's ViewerWindow wiring: the Trees/Eye candy
checkboxes, the composite-undo path Draw and Paint Can both feed into, the
Paint Can large-fill confirm guard, and settings persistence. Same offscreen
technique tests/test_fill_tool.py documents; every ViewerWindow() here must
call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import pytest

import conftest
from descape import settings
from descape.edit_history import CompositeDiffRecord, TileDiffRecord
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FOREST_OAK = 10  # density 1000/1000 -- deterministic single tree per tile
FOREST_OAK_CONST = 411


def _window(tool: str = "draw"):
    """The blank template loaded, Terrain mode, `tool` active, FOREST_OAK
    selected on terrain_combo. Caller must edit_history.mark_saved() +
    close()."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected(tool)
    window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(FOREST_OAK))
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
    import descape.viewer as viewer_module
    from PyQt5.QtWidgets import QMessageBox

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


def test_checkbox_visibility_follows_the_terrain_combo() -> None:
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
