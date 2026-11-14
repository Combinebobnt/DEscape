"""Units mode: Scatter, placing new units and moving the selection."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import QRectF
from unit_edit_viewer_helpers import (
    _BERRY_BUSH,
    _chunk_for_tile,
    _close,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _rebuilt_canvas,
    _select_keys,
    _whole_canvas,
    _window,
    _window_with_style,
)

from descape import render_units, settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# --- Edit > Scatter Units in Region… ----------------------------------------
#
# The dialog itself is tested headless in test_scatter_dialog.py; these drive
# the viewer around it, with exec_() monkeypatched so no modal box blocks the
# offscreen run.

_FISH_CONST = 456  # Fish (Salmon), the motivating GAIA case


_WATER_TERRAIN = 1  # TerrainId.WATER_SHALLOW


_POND = (60, 60, 70, 70)


def _paint_water(window, region) -> None:
    """The fixture is all grass, so a Water-only scatter needs a pond made
    first. Straight onto mm.terrain, the same flat index scatter_dialog
    reads."""
    mm = window.scenario.map_manager
    tx0, ty0, tx1, ty1 = region
    for y in range(ty0, ty1):
        for x in range(tx0, tx1):
            mm.terrain[y * mm.map_width + x].terrain_id = _WATER_TERRAIN


def _arm_scatter(window, monkeypatch, configure=None, accept=True):
    """Selects a region, picks the fish as GAIA, and auto-answers the dialog."""
    from PyQt5.QtWidgets import QDialog

    from descape.scatter_dialog import ScatterDialog

    window.units_panel.select_owner(0)
    window.units_panel.select_object(_FISH_CONST)
    window.on_region_selected(_POND)

    def fake_exec(dialog):
        if configure is not None:
            configure(dialog)
        return QDialog.Accepted if accept else QDialog.Rejected

    monkeypatch.setattr(ScatterDialog, "exec_", fake_exec)


def _water_40(dialog) -> None:
    dialog.water_radio.setChecked(True)
    dialog.count_spin.setValue(40)
    dialog.seed_spin.setValue(20260921)


def _gaia_fish(window) -> list:
    return [u for u in window.scenario.unit_manager.units[0] if u.unit_const == _FISH_CONST]


def test_the_scatter_action_is_gated_on_a_region_or_a_units_selection(monkeypatch) -> None:
    """GH #196: a Units-mode selection is an area too; the tooltip says how to get one."""
    from descape.viewer_common import _SCATTER_TIP
    from descape.viewer_tool_state import _SCATTER_NO_AREA_TIP

    window = _window()
    try:
        assert window._region is None and window._selection == []
        assert not window.scatter_action.isEnabled()
        assert window.scatter_action.toolTip() == _SCATTER_NO_AREA_TIP
        _select_keys(window, [(2, 300)])
        assert window.scatter_action.isEnabled()
        assert window.scatter_action.toolTip() == _SCATTER_TIP
        window.mode_combo.setCurrentText("Terrain")  # keeps the selection, live there (TASK-359)
        assert window.scatter_action.isEnabled()
        window.mode_combo.setCurrentText("View")  # keeps it dormant
        assert not window.scatter_action.isEnabled()
        assert window.scatter_action.toolTip() == _SCATTER_NO_AREA_TIP
        window.mode_combo.setCurrentText("Terrain")
        window.on_region_selected(_POND)
        window.mode_combo.setCurrentText("View")
        assert window.scatter_action.isEnabled()  # a region, in View mode
        window.mode_combo.setCurrentText("Units")
        assert window.scatter_action.isEnabled()  # and Units mode
        assert window.scatter_action.toolTip() == _SCATTER_TIP
        window.deselect_units()
        window.deselect()
        assert not window.scatter_action.isEnabled()
    finally:
        _close(window)


def test_the_scatter_action_is_disabled_with_no_map() -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow
    from descape.viewer_common import _SCATTER_TIP

    window = ViewerWindow()
    try:
        assert window.scenario is None
        assert not window.scatter_action.isEnabled()
        assert window.scatter_action.toolTip() == _SCATTER_TIP
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_scatter_with_no_object_chosen_opens_no_dialog(monkeypatch) -> None:
    window = _window()
    try:
        from PyQt5.QtWidgets import QDialog

        from descape.scatter_dialog import ScatterDialog

        opened = []

        def record_exec(dialog):
            opened.append(dialog)
            return QDialog.Rejected

        monkeypatch.setattr(ScatterDialog, "exec_", record_exec)
        window.on_region_selected(_POND)
        assert window.units_panel.selected_object_const() is None
        window.scatter_units_in_region()
        assert opened == []
        assert "choose an object in the Units panel first" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_scatter_places_every_unit_inside_the_region_and_on_water(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, _POND)
        _arm_scatter(window, monkeypatch, _water_40)
        before_records = len(window.edit_history.records)
        before_fish = len(_gaia_fish(window))

        window.scatter_units_in_region()

        fish = _gaia_fish(window)
        placed = len(fish) - before_fish
        # TASK-277: a 2x2 picked on the pond's far edge would overhang it, so it is skipped.
        assert 0 < placed < 40
        mm = window.scenario.map_manager
        for unit in fish:
            # GH #301: a 2x2 deep fish snaps to the corner one past its picked tile.
            assert unit.x == int(unit.x) and unit.y == int(unit.y)
            x0, x1, y0, y1 = render_units.unit_tile_bounds(unit, mm.map_width, mm.map_height)
            assert x0 >= 60 and x1 <= 70 and y0 >= 60 and y1 <= 70
            for ty in range(y0, y1):
                for tx in range(x0, x1):
                    assert mm.terrain[ty * mm.map_width + tx].terrain_id == _WATER_TERRAIN
            # AGENTS.md's hard rule: rotation is a variant index for most
            # GAIA objects, so scatter never invents one.
            assert unit.rotation == 0.0
            assert unit.initial_animation_frame == 0
        assert len({(int(u.x), int(u.y)) for u in fish}) == len(fish)  # one per tile
        assert len(window._selection) == placed
        assert set(window._selection) == {(0, u.reference_id) for u in fish}
        assert len(window.edit_history.records) == before_records + 1
        log = window.status_log.toPlainText()
        assert f"Scattered {placed} x" in log
        assert "asked for 40; clamped by eligible tiles, spacing or object size" in log

        window.undo()
        assert len(_gaia_fish(window)) == before_fish
    finally:
        _close(window)


def test_the_same_seed_scatters_to_the_same_coordinates(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, _POND)
        _arm_scatter(window, monkeypatch, _water_40)
        window.scatter_units_in_region()
        first = sorted((u.x, u.y) for u in _gaia_fish(window))
        window.undo()
        window.scatter_units_in_region()
        assert sorted((u.x, u.y) for u in _gaia_fish(window)) == first
    finally:
        _close(window)


def test_villager_random_scatters_a_seeded_mix_of_both_villagers(monkeypatch) -> None:
    """GH #164: each scattered copy rolls, from the dialog's seed."""
    from PyQt5.QtWidgets import QDialog

    from descape.place_random import RANDOM_VILLAGER
    from descape.scatter_dialog import ScatterDialog

    window = _window()
    try:
        window.units_panel.catalog_view.select(RANDOM_VILLAGER)
        window.units_panel.select_owner(2)
        window.on_region_selected(_POND)
        labels = []

        def fake_exec(dialog):
            labels.append(dialog.object_label.text())
            dialog.count_spin.setValue(40)
            dialog.seed_spin.setValue(20261011)
            return QDialog.Accepted

        monkeypatch.setattr(ScatterDialog, "exec_", fake_exec)
        before = len(window.scenario.unit_manager.units[2])

        def villagers():
            return [u.unit_const for u in window.scenario.unit_manager.units[2][before:]]

        window.scatter_units_in_region()
        first = villagers()
        assert len(first) == 40
        assert set(first) == {83, 293}
        assert labels == ["Object: Villager (Random)"]
        assert "Scattered 40 x Villager (Random) for Player 2" in window.status_log.toPlainText()

        window.undo()
        window.scatter_units_in_region()
        assert villagers() == first, "the same seed rolls the same mix"
    finally:
        _close(window)


def test_cancelling_the_scatter_dialog_records_nothing(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, _POND)
        _arm_scatter(window, monkeypatch, _water_40, accept=False)
        before_records = len(window.edit_history.records)
        before_fish = len(_gaia_fish(window))

        window.scatter_units_in_region()

        assert len(window.edit_history.records) == before_records
        assert len(_gaia_fish(window)) == before_fish
        # The dialog runs before _ensure_unit_edits(), so a cancel never pays
        # for the lazy model build.
        assert window.unit_edits is None
    finally:
        _close(window)


def test_only_an_accepted_scatter_is_remembered_for_the_next_dialog(monkeypatch) -> None:
    """TASK-514 pins Scatter's existing rule, which the guide now states."""
    window = _window()
    try:
        opened = []

        def set_count(count):
            def configure(dialog):
                opened.append(dialog.count_spin.value())
                dialog.count_spin.setValue(count)

            return configure

        _arm_scatter(window, monkeypatch, set_count(7))
        window.scatter_units_in_region()
        _arm_scatter(window, monkeypatch, set_count(23), accept=False)
        window.scatter_units_in_region()
        _arm_scatter(window, monkeypatch, set_count(7), accept=False)
        window.scatter_units_in_region()
        assert opened[1:] == [7, 7], "the accepted count opens next, a cancelled one is dropped"
        assert opened[0] != 7
    finally:
        _close(window)


def test_asking_for_more_fish_than_the_pond_holds_says_so(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, (60, 60, 63, 63))  # 9 water tiles inside the region

        def configure(dialog):
            dialog.water_radio.setChecked(True)
            dialog.count_spin.setValue(50)

        _arm_scatter(window, monkeypatch, configure)
        window.scatter_units_in_region()
        placed = len(_gaia_fish(window))
        # TASK-277: of 9 water tiles, only picks whose 2x2 stays on the water place.
        assert 0 < placed < 9
        log = window.status_log.toPlainText()
        assert f"Scattered {placed} x" in log
        assert "asked for 50" in log
    finally:
        _close(window)


def test_a_scatter_where_no_footprint_fits_records_nothing(monkeypatch) -> None:
    """TASK-277: every 2x2 picked on a one-tile-wide channel overhangs it, so
    nothing is placed and no empty undo step, dirty flag or selection change follows."""
    window = _window()
    try:
        _paint_water(window, (60, 60, 70, 61))

        def configure(dialog):
            dialog.water_radio.setChecked(True)
            dialog.count_spin.setValue(5)

        _arm_scatter(window, monkeypatch, configure)
        _select_keys(window, [(2, 300)])
        before_records = len(window.edit_history.records)
        before_fish = len(_gaia_fish(window))
        window.scatter_units_in_region()
        assert len(_gaia_fish(window)) == before_fish
        assert len(window.edit_history.records) == before_records
        assert window._selection == [(2, 300)]
        assert "fits the eligible tiles, so nothing was placed" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_scatter_outside_units_mode_leaves_the_selection_alone(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, _POND)
        _arm_scatter(window, monkeypatch, _water_40)
        window.mode_combo.setCurrentText("Terrain")
        window._selection = []
        window.scatter_units_in_region()
        assert _gaia_fish(window)
        assert window._selection == []
    finally:
        _close(window)


def test_scatter_pixels_appear_and_one_undo_restores_them(monkeypatch) -> None:
    """The cache-invalidation check b1 and b2 both needed: a model-only
    assertion passes while the map keeps painting the pre-edit chunk."""
    window = _window()
    try:
        _arm_scatter(window, monkeypatch, lambda dialog: dialog.count_spin.setValue(40))
        cx, cy = _chunk_for_tile(window, 65, 65)

        before = window._cache.get_chunk(0, cx, cy).copy()
        window.scatter_units_in_region()
        after = window._cache.get_chunk(0, cx, cy).copy()
        assert not np.array_equal(before, after)

        window.undo()
        restored = window._cache.get_chunk(0, cx, cy)
        assert np.array_equal(before, restored)
    finally:
        _close(window)


def _press_on_map(window, key) -> None:
    """A real Ctrl+<key> on the map of the activated, shown window, so the Edit menu keybind fires."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window.activateWindow()
    QApplication.setActiveWindow(window)
    window.map_view.setFocus()
    QApplication.processEvents()
    assert window.isActiveWindow() and QApplication.focusWidget() is window.map_view
    QTest.keyClick(window.map_view, key, Qt.ControlModifier)
    QTest.keyRelease(window.map_view, Qt.Key_Control)
    QApplication.processEvents()


def test_a_real_ctrl_z_takes_a_scatter_back_and_ctrl_y_puts_it_back(monkeypatch) -> None:
    """TASK-106: the keybinds, not window.undo()/redo().

    Passes when, on a shown window, a scatter adds GAIA units and changes
    the chunk under the region; a real Ctrl+Z on the map then leaves exactly
    the pre-scatter unit list and repaints that chunk to its pre-scatter
    pixels; and a real Ctrl+Y brings back the same units at the same
    coordinates and the scattered pixels."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.show()
        QApplication.processEvents()
        _arm_scatter(window, monkeypatch, lambda dialog: dialog.count_spin.setValue(40))
        cx, cy = _chunk_for_tile(window, 65, 65)
        gaia = window.scenario.unit_manager.units[0]
        before_units = [(u.reference_id, u.x, u.y) for u in gaia]
        before = window._cache.get_chunk(0, cx, cy).copy()

        window.scatter_units_in_region()
        scattered_units = [(u.reference_id, u.x, u.y) for u in window.scenario.unit_manager.units[0]]
        assert len(scattered_units) > len(before_units)
        scattered = window._cache.get_chunk(0, cx, cy).copy()
        assert not np.array_equal(before, scattered)

        _press_on_map(window, Qt.Key_Z)
        assert [(u.reference_id, u.x, u.y) for u in window.scenario.unit_manager.units[0]] == before_units
        assert np.array_equal(window._cache.get_chunk(0, cx, cy), before)

        _press_on_map(window, Qt.Key_Y)
        assert [(u.reference_id, u.x, u.y) for u in window.scenario.unit_manager.units[0]] == scattered_units
        assert np.array_equal(window._cache.get_chunk(0, cx, cy), scattered)
    finally:
        _close(window)


def test_scattering_a_gaia_only_object_for_a_player_places_it_for_gaia(monkeypatch) -> None:
    """TASK-288: Scatter of a Forage Bush with Player 1 picked lands on GAIA
    (Keep listed, GH #32: in hide mode Player 1 drops the pick)."""
    settings.set_gaia_only_catalog_mode("keep")
    window = _window()
    try:

        def configure(dialog):
            dialog.count_spin.setValue(12)
            dialog.seed_spin.setValue(20261011)

        _arm_scatter(window, monkeypatch, configure)
        window.units_panel.select_object(_BERRY_BUSH)
        window.units_panel.select_owner(1)
        before = {p: len(units) for p, units in enumerate(window.scenario.unit_manager.units)}

        window.scatter_units_in_region()

        after = {p: len(units) for p, units in enumerate(window.scenario.unit_manager.units)}
        assert after[0] == before[0] + 12
        assert {p: n for p, n in after.items() if p} == {p: n for p, n in before.items() if p}
        assert all(u.unit_const == _BERRY_BUSH for u in window.scenario.unit_manager.units[0][-12:])
        assert {key[0] for key in window._selection} == {0}
        status = window.status_log.toPlainText().splitlines()[-1]
        assert status.startswith("Scattered 12 x ")
        assert "for GAIA" in status and status.endswith("(GAIA-only, so placed for GAIA)")
        assert window.units_panel.owner_id() == 1
    finally:
        _close(window)


# --- GH #105: Scatter's "Move the selected units" mode -----------------------

_UNIT_FIELDS = (
    "x",
    "y",
    "z",
    "rotation",
    "unit_const",
    "initial_animation_frame",
    "status",
    "garrisoned_in_id",
    "reference_id",
    "caption_string_id",
)


def _all_fields(window) -> dict:
    """Every unit's every field, keyed by (player, reference_id)."""
    return {
        (p, u.reference_id): {f: getattr(u, f) for f in _UNIT_FIELDS}
        for p, units in enumerate(window.scenario.unit_manager.units)
        for u in units
    }


def _unit_by_ref(window, player: int, ref: int):
    return next(u for u in window.scenario.unit_manager.units[player] if u.reference_id == ref)


def _move_mode(seed: int = 20261004):
    def configure(dialog) -> None:
        dialog.move_radio.setChecked(True)
        dialog.seed_spin.setValue(seed)

    return configure


def _arm_move(window, monkeypatch, keys, seed: int = 20261004, region=_POND) -> None:
    _arm_scatter(window, monkeypatch, _move_mode(seed))
    window.on_region_selected(region)
    _select_keys(window, keys)


def _in_region(unit, region) -> bool:
    """Every tile of the unit's footprint lies inside the half-open region."""
    tx0, ty0, tx1, ty1 = region
    x0, x1, y0, y1 = render_units.unit_tile_bounds(unit, 120, 120)
    return tx0 <= x0 and x1 <= tx1 and ty0 <= y0 and y1 <= ty1


# Everything the fixture has outside player 1's archer, which stays unselected.
_SCATTER_KEYS = [(0, 100), (0, 101), (0, 102), (1, 200), (2, 300), (2, 301)]


def test_scatter_move_changes_only_the_selections_x_y_in_one_undo_step(monkeypatch) -> None:
    window = _window()
    try:
        _arm_move(window, monkeypatch, _SCATTER_KEYS)
        _unit_by_ref(window, 2, 300).z = 2.5  # every fixture z is 0.0, which would hide a zeroed z
        before = _all_fields(window)
        assert before[(0, 100)]["rotation"] == 7.0  # the oak's variant index, not an angle
        records = len(window.edit_history.records)

        window.scatter_units_in_region()

        after = _all_fields(window)
        assert len(window.edit_history.records) == records + 1
        assert "Scattered 6 selected units across the area" in window.status_log.toPlainText()
        # The house's garrisoned villager (203) rides to the house's new point (GH #42).
        house = _unit_by_ref(window, 1, 200)
        assert (after[(1, 203)]["x"], after[(1, 203)]["y"]) == (house.x, house.y)
        for key in before:
            moved = key in _SCATTER_KEYS or key == (1, 203)
            same = {f: v for f, v in before[key].items() if not (moved and f in ("x", "y"))}
            assert {f: after[key][f] for f in same} == same, key
            if key in _SCATTER_KEYS:
                assert _in_region(_unit_by_ref(window, *key), _POND), key
                assert (after[key]["x"], after[key]["y"]) != (before[key]["x"], before[key]["y"]), key
        # The house snaps to its 2x2 span on whole tiles.
        assert house.x % 1 == 0 and house.y % 1 == 0
        assert set(window._selection) == set(_SCATTER_KEYS)

        window.undo()
        assert _all_fields(window) == before
    finally:
        _close(window)


def test_scatter_move_gives_the_same_positions_for_the_same_seed_in_any_click_order(monkeypatch) -> None:
    window = _window()
    try:
        _arm_move(window, monkeypatch, _SCATTER_KEYS)
        window.scatter_units_in_region()
        first = _all_fields(window)
        window.undo()
        _select_keys(window, list(reversed(_SCATTER_KEYS)))
        window.scatter_units_in_region()
        assert _all_fields(window) == first
        window.undo()
        _arm_move(window, monkeypatch, _SCATTER_KEYS, seed=12345)
        window.scatter_units_in_region()
        assert _all_fields(window) != first
    finally:
        _close(window)


def test_scatter_move_with_nothing_selected_refuses(monkeypatch) -> None:
    window = _window()
    try:
        _arm_move(window, monkeypatch, [])
        before = _all_fields(window)
        records = len(window.edit_history.records)
        window.scatter_units_in_region()
        assert "select the units to move in Units mode first" in window.status_log.toPlainText()
        assert len(window.edit_history.records) == records
        assert _all_fields(window) == before
        assert window.unit_edits is None
    finally:
        _close(window)


def test_scatter_where_the_selection_is_dormant_is_handed_no_units_and_move_refuses(monkeypatch) -> None:
    """TASK-359: View keeps the selection but dormant, so a region's Scatter
    dialog gets no selected units and its move mode refuses."""
    from descape.scatter_dialog import ScatterDialog

    window = _window()
    try:
        _arm_move(window, monkeypatch, _SCATTER_KEYS)
        window.mode_combo.setCurrentText("View")
        assert sorted(window._selection) == sorted(_SCATTER_KEYS), "kept while dormant"
        answer = ScatterDialog.exec_
        handed = []

        def record_exec(dialog):
            handed.append(list(dialog._selected))
            return answer(dialog)

        monkeypatch.setattr(ScatterDialog, "exec_", record_exec)
        before = _all_fields(window)
        window.scatter_units_in_region()
        assert handed == [[]]
        assert "select the units to move in Units mode first" in window.status_log.toPlainText()
        assert _all_fields(window) == before
    finally:
        _close(window)


def test_scatter_move_in_terrain_moves_a_selection_made_in_units_mode(monkeypatch) -> None:
    """TASK-359: select units, go to Terrain and draw a region; move mode moves
    them in one undo step, and they stay selected with the highlight on them."""
    window = _window()
    try:
        mv = window.map_view
        _select_keys(window, _SCATTER_KEYS)
        window.mode_combo.setCurrentText("Terrain")
        _arm_scatter(window, monkeypatch, _move_mode())
        before = _all_fields(window)
        records = len(window.edit_history.records)

        window.scatter_units_in_region()

        assert len(window.edit_history.records) == records + 1
        assert "Scattered 6 selected units across the area" in window.status_log.toPlainText()
        for key in _SCATTER_KEYS:
            assert _in_region(_unit_by_ref(window, *key), _POND), key
        assert set(window._selection) == set(_SCATTER_KEYS)
        assert set(mv._unit_select_keys) == set(_SCATTER_KEYS)
        drawn, expected = QRectF(), QRectF()
        for _fill, _under, outline in mv._unit_select_groups.values():
            drawn = drawn.united(outline.path().boundingRect())
        for key in _SCATTER_KEYS:
            expected = expected.united(mv._unit_path(mv._unit_index.entry_for_key(key)).boundingRect())
        assert drawn == expected

        window.undo()
        assert _all_fields(window) == before
    finally:
        _close(window)


def test_scatter_with_no_region_and_nothing_selected_names_both_routes(monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    from descape.scatter_dialog import ScatterDialog

    window = _window()
    try:
        _arm_move(window, monkeypatch, [])
        window.on_region_selected(None)
        opened = []

        def record_exec(dialog):
            opened.append(dialog)
            return QDialog.Rejected

        monkeypatch.setattr(ScatterDialog, "exec_", record_exec)
        before = _all_fields(window)
        window.scatter_units_in_region()
        assert opened == []
        status = window.status_log.toPlainText().splitlines()[-1]
        assert status.startswith("Scatter: no area yet.")
        assert "Select in Terrain mode" in status and "select units in Units mode" in status
        assert _all_fields(window) == before
    finally:
        _close(window)


# --- GH #196: no region, so the area is the selected units' box -------------

# A 2x2 house (with its garrison) and two player-2 units: box (10, 10, 22, 21).
_BOX_KEYS = [(1, 200), (2, 300), (2, 301)]


def _arm_units_area(window, monkeypatch, keys, configure) -> None:
    """Fish as GAIA, no region, `keys` selected, and the dialog auto-answered."""
    from PyQt5.QtWidgets import QDialog

    from descape.scatter_dialog import ScatterDialog

    window.units_panel.select_owner(0)
    window.units_panel.select_object(_FISH_CONST)
    window.on_region_selected(None)
    _select_keys(window, keys)

    def fake_exec(dialog):
        configure(dialog)
        return QDialog.Accepted

    monkeypatch.setattr(ScatterDialog, "exec_", fake_exec)


def _keys_box(window, keys):
    from descape.scatter_dialog import units_region

    return units_region([_unit_by_ref(window, *k) for k in keys], 120, 120)


def _grow_move(grow: int, seed: int = 20261019, seen=None):
    def configure(dialog) -> None:
        if seen is not None:
            seen.append((dialog.units_radio.isChecked(), dialog.move_radio.isChecked(), dialog.region_radio.isEnabled()))
        dialog.grow_spin.setValue(grow)
        dialog.seed_spin.setValue(seed)

    return configure


@pytest.mark.parametrize("grow", [0, 3])
def test_scatter_with_no_region_moves_the_selection_inside_its_grown_box(monkeypatch, grow) -> None:
    from descape.scatter_dialog import grown_region

    window = _window()
    try:
        seen = []
        _arm_units_area(window, monkeypatch, _BOX_KEYS, _grow_move(grow, seen=seen))
        box = grown_region(_keys_box(window, _BOX_KEYS), grow)
        assert _keys_box(window, _BOX_KEYS) == (10, 10, 22, 21)
        before = _all_fields(window)
        records = len(window.edit_history.records)

        window.scatter_units_in_region()

        assert seen == [(True, True, False)], "units area, Move, no region to pick"
        after = _all_fields(window)
        assert len(window.edit_history.records) == records + 1
        assert "Scattered 3 selected units" in window.status_log.toPlainText()
        moved = set(_BOX_KEYS) | {(1, 203)}  # the house's garrison rides along
        for key in before:
            same = {f: v for f, v in before[key].items() if not (key in moved and f in ("x", "y"))}
            assert {f: after[key][f] for f in same} == same, key
        for key in _BOX_KEYS:
            assert _in_region(_unit_by_ref(window, *key), box), key
        assert after != before
        layout = after

        window.undo()
        assert _all_fields(window) == before
        _select_keys(window, _BOX_KEYS)
        window.scatter_units_in_region()
        assert _all_fields(window) == layout, "the same seed gives the same layout"
    finally:
        _close(window)


def test_scatter_with_no_region_places_new_units_around_the_selection(monkeypatch) -> None:
    from descape import scatter
    from descape.scatter_dialog import DEFAULT_GROW, grown_region

    def place_10(dialog) -> None:
        dialog.place_radio.setChecked(True)
        dialog.count_spin.setValue(10)
        dialog.seed_spin.setValue(20261019)

    window = _window()
    try:
        _arm_units_area(window, monkeypatch, _BOX_KEYS, place_10)
        box = grown_region(_keys_box(window, _BOX_KEYS), DEFAULT_GROW)
        occupied = scatter.occupied_tiles(window.scenario)
        before = len(_gaia_fish(window))

        window.scatter_units_in_region()

        fish = _gaia_fish(window)[before:]
        assert len(fish) == 10
        for unit in fish:
            assert _in_region(unit, box)
            assert (int(unit.x), int(unit.y)) not in occupied
        assert "Scattered 10 x" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_scatter_move_with_no_object_chosen_still_opens_in_move_mode(monkeypatch) -> None:
    window = _window()
    try:
        _arm_move(window, monkeypatch, [(2, 301)])
        monkeypatch.setattr(window.units_panel, "selected_object_const", lambda: None)
        window.scatter_units_in_region()
        assert _in_region(_unit_by_ref(window, 2, 301), _POND)
    finally:
        _close(window)


def test_a_building_that_cannot_fit_stays_put_and_the_rest_move(monkeypatch) -> None:
    window = _window()
    try:
        region = (60, 60, 61, 61)  # one tile: no room for the 2x2 house
        _arm_move(window, monkeypatch, [(1, 200), (2, 301)], region=region)
        before = _all_fields(window)
        window.scatter_units_in_region()
        after = _all_fields(window)
        assert after[(1, 200)] == before[(1, 200)]
        assert after[(1, 203)] == before[(1, 203)]  # its garrison stays with it
        assert (after[(2, 301)]["x"], after[(2, 301)]["y"]) == (60.5, 60.5)
        assert "1 did not fit and stayed put" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_only_buildings_that_cannot_fit_record_nothing(monkeypatch) -> None:
    window = _window()
    try:
        _arm_move(window, monkeypatch, [(1, 200)], region=(60, 60, 61, 61))
        records = len(window.edit_history.records)
        window.scatter_units_in_region()
        assert len(window.edit_history.records) == records
        assert "fits in the area; nothing moved" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_more_movers_than_tiles_share_tiles_and_say_so(monkeypatch) -> None:
    window = _window()
    try:
        keys = [(0, 100), (0, 101), (0, 102)]
        _arm_move(window, monkeypatch, keys, region=(60, 60, 62, 61))  # two tiles
        window.scatter_units_in_region()
        tiles = [(int(_unit_by_ref(window, *k).x), int(_unit_by_ref(window, *k).y)) for k in keys]
        assert set(tiles) == {(60, 60), (61, 60)}
        assert "1 share a tile: the area has too few free tiles" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_selected_occupant_rides_with_its_host(monkeypatch) -> None:
    window = _window()
    try:
        _arm_move(window, monkeypatch, [(1, 200), (1, 203)])
        window.scatter_units_in_region()
        house, villager = _unit_by_ref(window, 1, 200), _unit_by_ref(window, 1, 203)
        assert _in_region(house, _POND)
        assert (villager.x, villager.y) == (house.x, house.y)
    finally:
        _close(window)


def test_scatter_move_pixels_follow_the_unit_and_undo_restores_them(monkeypatch) -> None:
    window = _window()
    try:
        _arm_move(window, monkeypatch, [(2, 301)], region=(64, 64, 65, 65))
        cx, cy = _chunk_for_tile(window, 64, 64)
        before = window._cache.get_chunk(0, cx, cy).copy()
        window.scatter_units_in_region()
        assert not np.array_equal(before, window._cache.get_chunk(0, cx, cy))
        window.undo()
        assert np.array_equal(before, window._cache.get_chunk(0, cx, cy))
    finally:
        _close(window)


# --- TASK-031.18: a pruned Erase record undoes through splices ----------------


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_a_gaia_only_erase_keeps_gaia_and_undoes_through_splices(style: str, monkeypatch) -> None:
    """Erase captures all nine players at press. Pruned at commit to GAIA,
    the record takes _gaia_membership_diff(), so undo and redo of a mid-list
    removal splice (unpruned, the tail shape refused it and they repainted
    wholesale). Each repaint must match a wholesale rebuild of the same cache.
    Flat gets the diff too, but its cost ratio against this small fixture's
    unit count repaints wholesale anyway (_if_spliceable())."""
    scoped = style != "Flat"
    window = _window() if style == "Flat" else _window_with_style(style)
    try:
        window.eraser_action.setChecked(True)
        window.brush_size_spin.setValue(1)
        for name, action in window.eraser_target_actions.items():
            action.setChecked(name == "trees")
        gaia = window.scenario.unit_manager.units[0]
        assert [u.reference_id for u in gaia[:3]] == [100, 101, 102], "tree 101 must be mid-list"
        before = _whole_canvas(window)

        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(6, 5)], 0)
        window.on_edit_stroke_end()

        record = window.edit_history.peek_undo()
        assert record.label == "Erase object"
        assert set(record.before.players) == set(record.after.players) == {0}
        assert [u.reference_id for u in gaia[:2]] == [100, 102]
        undo_diff = window._membership_diff(record, undo=True)
        redo_diff = window._membership_diff(record, undo=False)
        assert undo_diff is not None and redo_diff is not None
        assert (len(undo_diff[0]), len(undo_diff[1])) == (0, 1)
        assert (len(redo_diff[0]), len(redo_diff[1])) == (1, 0)
        erased = _whole_canvas(window)
        assert not np.array_equal(before, erased), "erasing the tree should change the canvas"
        assert np.array_equal(erased, _rebuilt_canvas(window))

        changed = []
        real_after = window._after_unit_mutation
        monkeypatch.setattr(
            window, "_after_unit_mutation", lambda c=None, **k: (changed.append(c), real_after(c, **k))[1]
        )
        window.undo()
        assert [u.reference_id for u in gaia[:3]] == [100, 101, 102]
        assert bool(changed[-1]) is scoped, "the undo took the other repaint path"
        undone = _whole_canvas(window)
        assert np.array_equal(undone, before)
        assert np.array_equal(undone, _rebuilt_canvas(window))

        window.redo()
        assert bool(changed[-1]) is scoped, "the redo took the other repaint path"
        redone = _whole_canvas(window)
        assert np.array_equal(redone, erased)
        assert np.array_equal(redone, _rebuilt_canvas(window))
    finally:
        _close(window)
