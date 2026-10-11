"""GH #184's Eraser brush through a real offscreen ViewerWindow: the stroke
erases live, records one undo step per drag in Units and Terrain mode, takes
a host's garrison, and clears layered terrain in the same record.

Same fixture and Flat setup as tests/test_unit_edit_viewer.py. Its units:
GAIA trees 100/101 on (5, 5)/(6, 5) and a GAIA wall 102 on (7, 5); Player 1's
House 200 on (10, 10) holding Villager 203, an Archer 201 on (11, 10);
Player 2's Archer 300 on (20, 20) and Villager 301 on (21, 20).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import eraser

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _window(mode: str = "Units"):
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText(mode)
    window.eraser_action.setChecked(True)
    window.brush_size_spin.setValue(1)
    assert window._current_tool == "eraser"
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _refs(window) -> set[int]:
    return {u.reference_id for units in window.scenario.unit_manager.units for u in units}


def _tick(window, **targets) -> None:
    for name, action in window.eraser_target_actions.items():
        action.setChecked(targets.get(name, False))


def _stroke(window, *paths) -> None:
    """One drag: press, one MapView event per path, release."""
    window.on_edit_stroke_start()
    for path in paths:
        window.on_edit_stroke_tiles(path, 0)
    window.on_edit_stroke_end()


def _layered(window, tiles, layer: int = 3) -> list[int]:
    mm = window.scenario.map_manager
    indices = [y * mm.map_width + x for x, y in tiles]
    for i in indices:
        mm.terrain[i].layer = layer
    return indices


def test_a_units_mode_stroke_erases_live_and_records_one_undo_step() -> None:
    window = _window()
    try:
        before = _refs(window)
        records = len(window.edit_history.records)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(20, 20)], 0)
        # Removed before mouse-up, not batched to it.
        assert 300 not in _refs(window)
        window.on_edit_stroke_tiles([(21, 20)], 0)
        window.on_edit_stroke_end()

        assert _refs(window) == before - {300, 301}
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.records[-1].label == "Erase 2 objects"
        assert window.unit_edits._pending is None
        assert "Erased 2 object(s)" in window.status_log.toPlainText()
        # The pick index no longer offers the erased units.
        assert all(e.unit.reference_id not in (300, 301) for e in window.map_view._unit_index.entries)

        window.undo()
        assert _refs(window) == before
        window.redo()
        assert _refs(window) == before - {300, 301}
        assert len(window.edit_history.records) == records + 1
    finally:
        _close(window)


def test_a_stroke_that_erases_nothing_records_nothing() -> None:
    window = _window()
    try:
        records = len(window.edit_history.records)
        before = _refs(window)
        _stroke(window, [(60, 60), (61, 60)])
        # GAIA trees match neither default target.
        _stroke(window, [(5, 5), (6, 5)])
        assert _refs(window) == before
        assert len(window.edit_history.records) == records
        assert window.unit_edits._pending is None
        assert not window.edit_history.in_stroke
    finally:
        _close(window)


def test_unticked_targets_and_filter_hidden_units_survive() -> None:
    window = _window()
    try:
        _tick(window, trees=True)
        window.show_trees_action.setChecked(False)
        _stroke(window, [(5, 5), (6, 5), (20, 20)])
        assert {100, 101, 300} <= _refs(window)
        window.show_trees_action.setChecked(True)
        _stroke(window, [(5, 5), (6, 5), (7, 5)])
        refs = _refs(window)
        assert {100, 101} & refs == set()
        assert 102 in refs  # the wall is a Building, not ticked
    finally:
        _close(window)


def test_erasing_a_host_takes_its_hidden_garrison_in_one_undo_step() -> None:
    window = _window()
    try:
        window.show_garrisoned_action.setChecked(False)
        _tick(window, buildings=True)
        before = _refs(window)
        records = len(window.edit_history.records)
        _stroke(window, [(10, 10)])

        assert _refs(window) == before - {200, 203}
        assert len(window.edit_history.records) == records + 1
        assert "(1 garrisoned)" in window.status_log.toPlainText()
        window.undo()
        assert _refs(window) == before
    finally:
        _close(window)


def test_a_terrain_mode_stroke_erases_without_a_unit_index() -> None:
    window = _window("Terrain")
    try:
        assert window.eraser_action.isVisible() and window.eraser_action.isEnabled()
        assert window.map_view._unit_index is None
        _tick(window, trees=True)
        records = len(window.edit_history.records)
        _stroke(window, [(5, 5), (6, 5), (7, 5)])
        refs = _refs(window)
        assert {100, 101} & refs == set()
        assert 102 in refs
        assert len(window.edit_history.records) == records + 1
        window.undo()
        assert {100, 101} <= _refs(window)
    finally:
        _close(window)


def test_terrain_overlay_clears_layer_and_undo_restores_it() -> None:
    window = _window("Terrain")
    try:
        _tick(window, overlay=True)
        terrain = window.scenario.map_manager.terrain
        indices = _layered(window, [(40, 40), (41, 40)])
        records = len(window.edit_history.records)
        _stroke(window, [(40, 40), (41, 40), (42, 40)])

        assert [terrain[i].layer for i in indices] == [-1, -1]
        assert len(window.edit_history.records) == records + 1
        assert "cleared layered terrain on 2 tile(s)" in window.status_log.toPlainText()
        window.undo()
        assert [terrain[i].layer for i in indices] == [3, 3]
        window.redo()
        assert [terrain[i].layer for i in indices] == [-1, -1]
    finally:
        _close(window)


def test_overlay_and_units_in_one_drag_are_one_undo_step() -> None:
    window = _window()
    try:
        _tick(window, units=True, overlay=True)
        terrain = window.scenario.map_manager.terrain
        (index,) = _layered(window, [(20, 20)])
        before = _refs(window)
        records = len(window.edit_history.records)
        _stroke(window, [(20, 20)])

        assert terrain[index].layer == -1
        assert 300 not in _refs(window)
        assert len(window.edit_history.records) == records + 1
        window.undo()
        assert terrain[index].layer == 3
        assert _refs(window) == before
    finally:
        _close(window)


def test_overlay_is_disabled_and_ignored_without_a_layer_field() -> None:
    window = _window("Terrain")
    try:
        overlay = window.eraser_target_actions["overlay"]
        assert overlay.isEnabled()
        window.scenario.terrain_has_layer = False
        window._update_tool_enabled()
        assert not overlay.isEnabled()

        _tick(window, overlay=True)
        assert not window._erase_targets().overlay
        (index,) = _layered(window, [(40, 40)])
        records = len(window.edit_history.records)
        _stroke(window, [(40, 40)])
        assert window.scenario.map_manager.terrain[index].layer == 3
        assert len(window.edit_history.records) == records
    finally:
        _close(window)


def test_everything_ticks_six_targets_but_not_cliffs() -> None:
    window = _window()
    try:
        _tick(window)
        menu = window.eraser_targets_button.menu()
        (everything,) = [a for a in menu.actions() if a.text() == "Everything"]
        everything.trigger()
        ticked = {name for name, action in window.eraser_target_actions.items() if action.isChecked()}
        assert ticked == eraser.EVERYTHING_FIELDS
        assert "cliffs" not in ticked
    finally:
        _close(window)


@pytest.mark.parametrize("mode", ["Units", "Terrain"])
def test_the_targets_menu_and_brush_show_only_for_the_eraser(mode: str) -> None:
    window = _window(mode)
    try:
        assert window.eraser_targets_param_action.isVisible()
        assert window.eraser_targets_button.isEnabled()
        assert window.brush_size_spin_action.isVisible()
        assert window.tool_param_separator_action.isVisible()
        window.pan_action.setChecked(True)
        assert not window.eraser_targets_param_action.isVisible()
        assert not window.eraser_targets_button.isEnabled()
    finally:
        _close(window)


def test_the_eraser_hides_in_view_mode() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("View")
        assert not window.eraser_action.isVisible()
        assert window._current_tool == "pan"
    finally:
        _close(window)


def test_the_stroke_end_closes_an_open_erase_stroke_whatever_the_tool() -> None:
    window = _window()
    try:
        _tick(window, units=True, overlay=True)
        _layered(window, [(20, 20)])
        records = len(window.edit_history.records)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(20, 20)], 0)
        window._current_tool = "pan"  # routing reads the open stroke, not the tool
        window.on_edit_stroke_end()
        assert window._erase is None
        assert not window.edit_history.in_stroke
        assert window.unit_edits._pending is None
        assert len(window.edit_history.records) == records + 1
    finally:
        _close(window)


def test_a_raise_mid_stroke_still_closes_both_edits(monkeypatch) -> None:
    window = _window()
    try:
        _tick(window, units=True, overlay=True)
        _layered(window, [(20, 20), (21, 20)])
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(20, 20)], 0)

        def boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(window, "_erase_units", boom)
        with pytest.raises(RuntimeError, match="boom"):
            window.on_edit_stroke_tiles([(21, 20)], 0)
        assert window._erase is None
        assert not window.edit_history.in_stroke
        assert window.unit_edits._pending is None
        window.on_edit_stroke_end()  # MapView still sends the release
        assert 300 not in _refs(window)
        window.undo()
        assert 300 in _refs(window)
    finally:
        _close(window)


def test_trigger_references_are_warned_not_refused(monkeypatch) -> None:
    window = _window()
    try:
        monkeypatch.setattr(window, "_find_trigger_ref_counts", lambda: {300: 2})
        _stroke(window, [(20, 20), (21, 20)])
        assert {300, 301} & _refs(window) == set()
        assert "Erased 2 object(s); 1 referenced by triggers" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_the_mid_stroke_flush_repaints_before_release() -> None:
    import numpy as np

    window = _window()
    try:
        tp = window.map_view._tile_pixels
        cp = window._cache.chunk_px
        cx, cy = (20 * tp) // cp, (20 * tp) // cp
        before = window._cache.get_chunk(0, cx, cy).copy()
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(20, 20)], 0)
        assert window._erase_refresh_timer.isActive()
        window._erase_refresh_timer.stop()
        window._flush_erase_refresh()
        mid = window._cache.get_chunk(0, cx, cy).copy()
        assert window._erase is not None  # still mid-stroke
        window.on_edit_stroke_end()
        assert not np.array_equal(before, mid)
    finally:
        _close(window)
