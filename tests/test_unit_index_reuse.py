"""The Units-mode pick index is current or None in every mode, so entering
Units mode can reuse it instead of rebuilding (plan 2026-09-30 units mode
switch). Each test compares against a fresh unit_pick.build_index(), never a
stored snapshot."""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import settings, unit_pick
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TOWN_CENTRE = 109
_TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def _enter_units(window) -> None:
    from PyQt5.QtWidgets import QApplication

    window.mode_combo.setCurrentText("Units")
    QApplication.processEvents()
    assert window.mode == "units"


def _units_window(path=BLANK_TEMPLATE_PATH):
    """Outlines left at their default, off, so nothing outside Units needs the index."""
    window = conftest.stepped_window(path)
    assert settings.get_footprint_outlines() is False
    _enter_units(window)
    return window


def _place_town_centre(window, tile=(50, 50)):
    """A Place in Units mode (patches the index in place); returns the unit's key."""
    count = sum(len(units) for units in window.scenario.unit_manager.units)
    window.units_panel.select_object(_TOWN_CENTRE)
    window.on_unit_place(window.map_view._tile_polygon(*tile).boundingRect().center(), None)
    assert sum(len(units) for units in window.scenario.unit_manager.units) == count + 1, "the place did not happen"
    (key,) = window._selection
    return key


def _assert_index_is_fresh(window) -> None:
    index = window.map_view._unit_index
    assert index is not None
    fresh = unit_pick.build_index(window.scenario, window._unit_filter)
    assert set(index.by_key) == set(fresh.by_key)
    assert index.by_tile == fresh.by_tile


def test_reentering_units_mode_with_no_edits_reuses_the_index(monkeypatch) -> None:
    """A call-counting fake, not monkeypatch.undo(), which would also revert the autouse fixtures."""
    calls = []
    real = unit_pick.build_index
    monkeypatch.setattr(unit_pick, "build_index", lambda *a, **k: calls.append(1) or real(*a, **k))
    window = _units_window()
    try:
        assert len(calls) == 1
        window.mode_combo.setCurrentText("Terrain")
        _enter_units(window)
        window.mode_combo.setCurrentText("Players")
        _enter_units(window)
        assert len(calls) == 1
        _assert_index_is_fresh(window)
    finally:
        conftest.close_window(window)


def test_undoing_a_place_outside_units_mode_drops_the_index_and_reentry_rebuilds_it() -> None:
    window = _units_window()
    try:
        key = _place_town_centre(window)
        window.mode_combo.setCurrentText("Terrain")
        assert window.map_view._unit_index is not None, "leaving Units mode keeps a current index"
        window.undo()
        assert window.map_view._unit_index is None
        _enter_units(window)
        _assert_index_is_fresh(window)
        assert window.map_view._unit_index.entry_for_key(key) is None
    finally:
        conftest.close_window(window)


def test_a_filter_change_outside_units_mode_drops_the_index_and_reentry_applies_it() -> None:
    window = _units_window()
    try:
        _place_town_centre(window)
        assert window.map_view._unit_index.entries
        window.mode_combo.setCurrentText("Terrain")
        window.filter_hide_all_action.trigger()
        assert window.map_view._unit_index is None
        _enter_units(window)
        _assert_index_is_fresh(window)
        assert window.map_view._unit_index.entries == []
    finally:
        conftest.close_window(window)


def test_the_trigger_picker_never_picks_against_a_unit_removed_outside_units_mode() -> None:
    """Pick from map rebuilds only a None index, so a stale one would still
    offer the removed unit."""
    from PyQt5.QtWidgets import QApplication

    gen = conftest.load_verify_module("gen_trigger_fixture")
    window = _units_window(_TRIGGER_FIXTURE)
    try:
        key = _place_town_centre(window)
        window.mode_combo.setCurrentText("Terrain")
        window.undo()
        window.mode_combo.setCurrentText("Triggers")
        QApplication.processEvents()
        window.trigger_panel.select_trigger(gen.REFERENCE_TRIGGER)
        window.arm_unit_picker((gen.REFERENCE_TRIGGER, "effect", gen.TASK_OBJECT_EFFECT, "selected_object_ids", True))
        assert window._unit_picker is not None
        assert window.map_view._unit_index.entry_for_key(key) is None
        _assert_index_is_fresh(window)
    finally:
        window.disarm_unit_picker()
        conftest.close_window(window)
