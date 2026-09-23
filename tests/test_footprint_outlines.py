"""View > Footprint Outlines: the scope predicate (Qt-free) and the scene
item MapView draws it with.

The overlay's geometry is unit_pick.unit_polygons(), the same function the
hover and selection cues use, so the tests here pin that it stays shared
rather than re-deriving spans of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from descape import render, settings, unit_pick
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

_TOWN_CENTRE = 109
_STONE_WALL = 117
_TREE = 349
_VILLAGER = 83
_HOUSE = 70


@dataclass
class _Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    reference_id: int = 1
    garrisoned_in_id: int = -1
    z: float = 0.0


def _index(*units) -> unit_pick.UnitIndex:
    index = unit_pick.UnitIndex()
    for order, unit in enumerate(units):
        bounds = render.unit_tile_bounds(unit, 120, 120)
        assert bounds is not None, unit
        unit.reference_id = 100 + order
        unit_pick._append_entry(index, 1, unit, bounds)
    return index


def _consts(entries) -> list[int]:
    return [entry.unit.unit_const for entry in entries]


# --- the scope predicate -----------------------------------------------------


def test_each_scope_selects_its_own_subset() -> None:
    """A 4x4 building, a 1x1 building and a non-building, which is what
    separates the three scopes from each other."""
    assert render.tile_span(_TOWN_CENTRE, render.NON_BUILDING_SPAN) > (1, 1)
    assert _STONE_WALL in render.BUILDING_TILE_SPANS
    assert _TREE not in render.BUILDING_TILE_SPANS
    index = _index(
        _Unit(10.0, 10.0, _TOWN_CENTRE), _Unit(20.5, 20.5, _STONE_WALL), _Unit(30.5, 30.5, _TREE)
    )
    assert _consts(unit_pick.footprint_entries(index, unit_pick.FOOTPRINT_SCOPE_MULTITILE)) == [_TOWN_CENTRE]
    assert _consts(unit_pick.footprint_entries(index, unit_pick.FOOTPRINT_SCOPE_BUILDINGS)) == [
        _TOWN_CENTRE,
        _STONE_WALL,
    ]
    assert _consts(unit_pick.footprint_entries(index, unit_pick.FOOTPRINT_SCOPE_ALL)) == [
        _TOWN_CENTRE,
        _STONE_WALL,
        _TREE,
    ]


def test_multitile_reads_the_shared_span_table_not_a_rule_of_its_own(monkeypatch) -> None:
    """The discriminating form: a const injected into the span table has to
    change the answer, which a hard-coded building list would not."""
    index = _index(_Unit(30.5, 30.5, _VILLAGER))
    assert unit_pick.footprint_entries(index, unit_pick.FOOTPRINT_SCOPE_MULTITILE) == []
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, _VILLAGER, (2, 2))
    assert _consts(unit_pick.footprint_entries(index, unit_pick.FOOTPRINT_SCOPE_MULTITILE)) == [_VILLAGER]


def test_an_unknown_scope_raises_rather_than_drawing_nothing() -> None:
    with pytest.raises(ValueError):
        unit_pick.footprint_entries(_index(_Unit(10.0, 10.0, _TOWN_CENTRE)), "everything")


def test_the_scope_default_is_a_real_scope() -> None:
    assert unit_pick.FOOTPRINT_SCOPE_DEFAULT in unit_pick.FOOTPRINT_SCOPES


# --- settings ----------------------------------------------------------------


def test_settings_default_off_at_the_default_scope(tmp_path) -> None:
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_footprint_outlines() is False
    assert settings.get_footprint_scope() == unit_pick.FOOTPRINT_SCOPE_DEFAULT


def test_settings_read_from_config(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text("footprint_outlines: true\nfootprint_scope: all\n")
    assert settings.get_footprint_outlines() is True
    assert settings.get_footprint_scope() == unit_pick.FOOTPRINT_SCOPE_ALL


def test_an_off_list_scope_falls_back(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text("footprint_scope: everything\n")
    assert settings.get_footprint_scope() == unit_pick.FOOTPRINT_SCOPE_DEFAULT


def test_settings_round_trip_through_the_file(monkeypatch) -> None:
    settings.set_footprint_outlines(True)
    settings.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_BUILDINGS)
    monkeypatch.setattr(settings, "_footprint_outlines", None)
    monkeypatch.setattr(settings, "_footprint_scope", None)
    assert settings.get_footprint_outlines() is True
    assert settings.get_footprint_scope() == unit_pick.FOOTPRINT_SCOPE_BUILDINGS


def test_setting_an_off_list_scope_raises_and_writes_nothing(tmp_path) -> None:
    with pytest.raises(ValueError):
        settings.set_footprint_scope("everything")
    assert not (tmp_path / "config.yaml").exists()


def test_the_keybind_row_ships_unbound_and_stays_with_the_view_rows() -> None:
    row = next(r for r in settings.REBINDABLE_ACTIONS if r[0] == "view_footprint_outlines")
    assert row[2] == ""
    ids = [action_id for action_id, _label, _key in settings.REBINDABLE_ACTIONS]
    view_positions = [i for i, action_id in enumerate(ids) if action_id.startswith("view_")]
    assert view_positions == list(range(view_positions[0], view_positions[-1] + 1))


# --- the overlay item --------------------------------------------------------

def _window_with_units(style: str = "Stepped"):
    """A window carrying a Town Centre, two houses sharing an edge and a
    villager, so scope, seams and geometry all have something to bite on."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    units = window.scenario.unit_manager.units
    units[1].append(_Unit(20.0, 20.0, _TOWN_CENTRE, reference_id=201))
    units[1].append(_Unit(30.0, 30.0, _HOUSE, reference_id=202))
    units[1].append(_Unit(32.0, 30.0, _HOUSE, reference_id=203))
    units[1].append(_Unit(40.5, 40.5, _VILLAGER, reference_id=204))
    units[1].append(_Unit(35.5, 35.5, _STONE_WALL, reference_id=205))
    window.terrain_style_combo.setCurrentText(style)
    window.refresh_map()
    QApplication.processEvents()
    window.footprint_action.setChecked(True)
    return window


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_the_overlay_traces_exactly_the_same_geometry_as_the_hover_cue(style: str) -> None:
    """One function feeds both today; this pins that it keeps doing so, in
    every style (their geometry differs, so a style-specific drift would
    otherwise be silent)."""
    window = _window_with_units(style)
    try:
        view = window.map_view
        view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_ALL)
        overlay = view._footprint_item.path()
        expected_points = set()
        for entry in view._unit_index.entries:
            for polygon in view._unit_polygons_for(entry) or []:
                expected_points |= {(round(x, 3), round(y, 3)) for x, y in polygon}
        drawn = {
            (round(overlay.elementAt(i).x, 3), round(overlay.elementAt(i).y, 3))
            for i in range(overlay.elementCount())
        }
        assert expected_points
        assert drawn == expected_points
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_every_subpath_is_closed() -> None:
    """A diamond stroked from an open subpath draws 3 edges of 4, which no
    polygon count would catch."""
    from PyQt5.QtGui import QPainterPath

    window = _window_with_units("Stepped")
    try:
        path = window.map_view._footprint_item.path()
        moves = sum(
            1 for i in range(path.elementCount()) if path.elementAt(i).type == QPainterPath.MoveToElement
        )
        # One MoveTo per subpath, and closeSubpath() adds the returning
        # LineTo, so a closed 4-gon is 5 elements: 1 move + 4 lines.
        assert moves > 0
        assert path.elementCount() == 5 * moves
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_the_scope_changes_what_is_drawn_and_persists() -> None:
    window = _window_with_units("Stepped")
    try:
        view = window.map_view
        counts = {}
        for scope in unit_pick.FOOTPRINT_SCOPES:
            window.footprint_scope_actions[scope].setChecked(True)
            assert settings.get_footprint_scope() == scope
            counts[scope] = view._footprint_item.path().elementCount()
        assert (
            counts[unit_pick.FOOTPRINT_SCOPE_MULTITILE]
            < counts[unit_pick.FOOTPRINT_SCOPE_BUILDINGS]
            < counts[unit_pick.FOOTPRINT_SCOPE_ALL]
        )
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_an_elevation_edit_resyncs_the_outlines() -> None:
    """Against freshly recomputed geometry, never a stored snapshot: a test
    that pinned the pre-edit points would pass with the bug present."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _window_with_units("Stepped")
    try:
        view = window.map_view
        before = view._footprint_item.path()
        assert before.elementCount() > 0
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(20, 20, Qt.NoModifier)
        window.on_edit_stroke_end()
        # The rebuild is deferred to the event loop, deliberately: see
        # MapView.refresh_elevation_overlays.
        QApplication.processEvents()

        after = view._footprint_item.path()
        expected = set()
        for entry in unit_pick.footprint_entries(view._unit_index, view._footprint_scope):
            for polygon in view._unit_polygons_for(entry) or []:
                expected |= {(round(x, 3), round(y, 3)) for x, y in polygon}
        drawn = {
            (round(after.elementAt(i).x, 3), round(after.elementAt(i).y, 3))
            for i in range(after.elementCount())
        }
        assert drawn == expected
        assert drawn != {
            (round(before.elementAt(i).x, 3), round(before.elementAt(i).y, 3))
            for i in range(before.elementCount())
        }, "the edit did not move any outline"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_filter_change_reaches_the_overlay() -> None:
    """The index is filter-aware, so a filtered-out unit must lose its
    outline without the overlay knowing what a filter is."""
    window = _window_with_units("Stepped")
    try:
        view = window.map_view
        view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_ALL)
        before = view._footprint_item.path().elementCount()
        window.filter_no_players_action.trigger()
        assert view._footprint_item.path().elementCount() < before
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_the_menu_wiring_persists_greys_never_and_writes_no_config_at_startup(tmp_path) -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window_with_units("Stepped")
    try:
        assert window.footprint_action.isChecked() is True  # the fixture turned it on
        assert settings.get_footprint_outlines() is True
        assert window.map_view._footprint_item.isVisible()
        window.footprint_action.setChecked(False)
        assert not window.map_view._footprint_item.isVisible()
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            assert window.footprint_action.isEnabled(), style
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_building_a_window_writes_no_config_file(tmp_path) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        assert not (tmp_path / "config.yaml").exists()
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_the_overlay_sits_below_the_hover_and_selection_cues() -> None:
    from descape.map_view import MapView

    window = _window_with_units("Stepped")
    try:
        assert MapView.FOOTPRINT_Z < MapView.UNIT_HOVER_Z < MapView.UNIT_SELECT_Z
        assert window.map_view._footprint_item.zValue() == MapView.FOOTPRINT_Z
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_closing_a_map_then_toggling_does_not_touch_a_deleted_item() -> None:
    window = _window_with_units("Stepped")
    try:
        window.map_view.clear_image()
        assert window.map_view._footprint_item is None
        window.map_view.set_footprint_outlines(True)
        window.map_view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_ALL)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_multi_touch_elevation_stroke_rebuilds_the_path_once() -> None:
    """Rebuilding per touched tile is ~1s per touch at All units in Sloped on
    the largest example file, so the rebuild is coalesced through the event
    loop instead."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _window_with_units("Stepped")
    try:
        view = window.map_view
        rebuilds = []
        real_refresh = view.refresh_footprint_overlay
        view.refresh_footprint_overlay = lambda: (rebuilds.append(1), real_refresh())[1]
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        for step in range(8):
            window.on_edit_stroke_tile(20 + step, 20, Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        assert len(rebuilds) == 1
    finally:
        view.refresh_footprint_overlay = real_refresh
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_placing_a_unit_updates_the_outlines_without_an_index_rebuild() -> None:
    """Place patches the index in place rather than rebuilding it, so
    everything derived from the index has to be refreshed alongside -- the
    stacks already were, the outlines are what this adds."""
    from PyQt5.QtWidgets import QApplication

    # The blank template's own units, not this module's duck-typed ones:
    # UnitEditModel edits the real scenario objects.
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        view = window.map_view
        window.footprint_action.setChecked(True)
        view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_ALL)
        window.mode_combo.setCurrentText("Units")
        window.show()
        QApplication.processEvents()
        before = view._footprint_item.path().elementCount()
        placed_before = len(window.scenario.unit_manager.units[1])
        window.units_panel.select_object(_TOWN_CENTRE)
        # on_unit_place takes a SCENE point (it calls _pick_tile directly).
        window.on_unit_place(view._tile_polygon(50, 50).boundingRect().center(), None)
        assert len(window.scenario.unit_manager.units[1]) == placed_before + 1, "the place did not happen"
        assert view._footprint_item.path().elementCount() > before
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_unit_edit_made_outside_units_mode_still_reaches_the_outlines() -> None:
    """Paste, mirror and undo all land in _after_unit_mutation from whatever
    mode is current, and the overlay is live in all of them."""
    window = _window_with_units("Stepped")
    try:
        view = window.map_view
        view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_ALL)
        before = view._footprint_item.path().elementCount()
        assert window.mode != "units"
        units = window.scenario.unit_manager.units
        units[1].append(_Unit(50.0, 50.0, _TOWN_CENTRE, reference_id=301))
        window._after_unit_mutation()
        assert view._footprint_item.path().elementCount() > before
    finally:
        conftest.close_window(window)


_FARM = 50


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_show_sprites_toggle_resyncs_draped_farm_outlines_in_sloped() -> None:
    """A Sloped farm drapes over its tiles only while sprites are on, so the
    outline changes shape on the toggle with no index or elevation event.
    Checked against freshly recomputed geometry both ways, on a ramp so the
    draped and diamond shapes cannot coincide."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        for tile in window.scenario.map_manager.terrain:
            tile.elevation = max(0, min(4, tile.x - 18))
        window.scenario.unit_manager.units[1].append(_Unit(21.5, 21.5, _FARM, reference_id=401))
        window.terrain_style_combo.setCurrentText("Sloped")
        window.refresh_map()
        QApplication.processEvents()
        window.footprint_action.setChecked(True)
        view = window.map_view
        view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_ALL)
        assert render._terrain_overlay_for(_FARM) is not None, "not a farm to the render path"

        def drawn():
            path = view._footprint_item.path()
            return {
                (round(path.elementAt(i).x, 3), round(path.elementAt(i).y, 3))
                for i in range(path.elementCount())
            }

        def expected():
            points = set()
            for entry in unit_pick.footprint_entries(view._unit_index, view._footprint_scope):
                for polygon in view._unit_polygons_for(entry) or []:
                    points |= {(round(x, 3), round(y, 3)) for x, y in polygon}
            return points

        assert window.show_sprites_action.isChecked(), "sprites ship on; this test toggles off first"
        before = drawn()
        assert before == expected() and before
        for on in (False, True):
            window.show_sprites_action.setChecked(on)
            cache = view._sloped_cache()
            assert cache is not None and cache.with_units and cache.sprites_enabled is on
            assert drawn() == expected(), f"sprites {'on' if on else 'off'}: outline is stale"
            if not on:
                assert drawn() != before, "the toggle did not change any outline"
        assert drawn() == before
    finally:
        conftest.close_window(window)
