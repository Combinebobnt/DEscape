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
_STONE_GATE = 659  # the diagonal "e" orientation: six tiles that touch only at corners
_TREBUCHET = 42
# BUILDING_TILE_SPANS members that Units only still outlines: mobile siege, plus 1192 (no placements).
_SIEGE_IN_BOTH_SCOPES = (42, 331, 444, 479, 682, 683, 729, 730, 1192, 1690, 1691)


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


def test_units_only_drops_buildings_walls_and_gates_and_keeps_siege() -> None:
    """GH #143: a building, a wall, a gate, a trebuchet and two ordinary units."""
    index = _index(
        _Unit(10.0, 10.0, _TOWN_CENTRE),
        _Unit(20.5, 20.5, _STONE_WALL),
        _Unit(30.0, 30.0, _STONE_GATE),
        _Unit(40.5, 40.5, _TREBUCHET),
        _Unit(50.5, 50.5, _VILLAGER),
        _Unit(60.5, 60.5, _TREE),
    )
    assert _consts(unit_pick.footprint_entries(index, unit_pick.FOOTPRINT_SCOPE_UNITS)) == [
        _TREBUCHET,
        _VILLAGER,
        _TREE,
    ]


def test_units_only_reuses_the_show_buildings_and_show_walls_sets(monkeypatch) -> None:
    """The discriminating form: shrinking building_consts() changes the answer."""
    from descape import unit_kind

    index = _index(_Unit(10.0, 10.0, _TOWN_CENTRE))
    assert unit_pick.footprint_entries(index, unit_pick.FOOTPRINT_SCOPE_UNITS) == []
    monkeypatch.setattr(unit_kind, "building_consts", frozenset)
    assert _consts(unit_pick.footprint_entries(index, unit_pick.FOOTPRINT_SCOPE_UNITS)) == [_TOWN_CENTRE]


def test_all_buildings_and_units_only_overlap_on_exactly_the_mobile_siege() -> None:
    from descape import unit_kind

    excluded = unit_kind.building_consts() | unit_kind.wall_consts()
    assert tuple(sorted(c for c in render.BUILDING_TILE_SPANS if c not in excluded)) == _SIEGE_IN_BOTH_SCOPES


def test_units_only_sits_between_all_buildings_and_all_units() -> None:
    scopes = unit_pick.FOOTPRINT_SCOPES
    assert scopes.index(unit_pick.FOOTPRINT_SCOPE_BUILDINGS) + 1 == scopes.index(unit_pick.FOOTPRINT_SCOPE_UNITS)
    assert scopes.index(unit_pick.FOOTPRINT_SCOPE_UNITS) + 1 == scopes.index(unit_pick.FOOTPRINT_SCOPE_ALL)


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


def test_merge_and_owner_colour_default_off(tmp_path) -> None:
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_footprint_merged() is False
    assert settings.get_footprint_by_owner() is False


def test_merge_and_owner_colour_round_trip_through_the_file(monkeypatch) -> None:
    settings.set_footprint_merged(True)
    settings.set_footprint_by_owner(True)
    monkeypatch.setattr(settings, "_footprint_merged", None)
    monkeypatch.setattr(settings, "_footprint_by_owner", None)
    assert settings.get_footprint_merged() is True
    assert settings.get_footprint_by_owner() is True
    settings.set_footprint_merged(False)
    monkeypatch.setattr(settings, "_footprint_merged", None)
    assert settings.get_footprint_merged() is False
    assert settings.get_footprint_by_owner() is True


def test_the_outline_colour_defaults_to_light_gray(tmp_path) -> None:
    """Not white: the hover cue (unit_hover) is #ffffff and draws over the outlines."""
    assert settings.get_overlay_color("footprint_outline") == "#c8c8c8"
    assert settings.get_overlay_color("unit_hover") != "#c8c8c8"


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


def _overlay_item(view):
    """The configured-colour group: always present while a map is loaded."""
    return view._footprint_items[None]


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
        overlay = _overlay_item(view).path()
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


def _subpath_count(path) -> int:
    from PyQt5.QtGui import QPainterPath

    return sum(1 for i in range(path.elementCount()) if path.elementAt(i).type == QPainterPath.MoveToElement)


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_merged_outlines_trace_a_subset_of_the_hover_cue_one_ring_per_building(style: str) -> None:
    """GH #143 Merge: no new vertex, the same extent, and each multi-tile
    building (Town Centre, two houses) one closed subpath instead of one per tile."""
    window = _window_with_units(style)
    try:
        view = window.map_view
        if style == "Flat":
            # Offscreen, a restyle never reaches MapView (GOTCHAS); safe for geometry-only checks.
            view._terrain_style = "flat"
        assert view._terrain_style == style.lower()
        view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_MULTITILE)
        per_tile = _overlay_item(view).path()
        window.footprint_merged_action.setChecked(True)
        merged = _overlay_item(view).path()
        assert _drawn_points(view) <= _expected_points(view)
        assert merged.boundingRect() == per_tile.boundingRect()
        assert _subpath_count(merged) == 3
        # Flat already draws one rect per unit, so Merge has nothing to join there.
        assert _subpath_count(per_tile) == (3 if style == "Flat" else 16 + 4 + 4)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_merge_keeps_a_diagonal_gate_per_tile_beside_a_merged_building() -> None:
    """The gate's six tiles touch only at corners (pinch vertices), so it falls
    back to its per-tile outlines while the Town Centre still merges."""
    window = _window_with_units("Stepped")
    try:
        view = window.map_view
        window.scenario.unit_manager.units[1].append(_Unit(60.0, 60.0, _STONE_GATE, reference_id=206))
        window._after_unit_mutation()
        window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_BUILDINGS].setChecked(True)
        window.footprint_merged_action.setChecked(True)
        (gate,) = [e for e in view._unit_index.entries if e.unit.unit_const == _STONE_GATE]
        gate_points = {(round(x, 3), round(y, 3)) for p in view._unit_polygons_for(gate) for x, y in p}
        assert len(view._unit_polygons_for(gate)) == 6
        assert gate_points <= _drawn_points(view)
        # Town Centre 1 + two houses 2 + wall 1 + the gate's 6 tiles.
        assert _subpath_count(_overlay_item(view).path()) == 1 + 2 + 1 + 6
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
        path = _overlay_item(window.map_view).path()
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
            counts[scope] = _overlay_item(view).path().elementCount()
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
        before = _overlay_item(view).path()
        assert before.elementCount() > 0
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(20, 20, Qt.NoModifier)
        window.on_edit_stroke_end()
        # The rebuild is deferred to the event loop, deliberately: see
        # MapView.refresh_elevation_overlays.
        QApplication.processEvents()

        after = _overlay_item(view).path()
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
        before = _overlay_item(view).path().elementCount()
        window.filter_no_players_action.trigger()
        assert _overlay_item(view).path().elementCount() < before
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
        assert _overlay_item(window.map_view).isVisible()
        view = window.map_view
        assert window.footprint_merged_action.isChecked() is False
        assert window.footprint_by_owner_action.isChecked() is False
        window.footprint_merged_action.setChecked(True)
        assert settings.get_footprint_merged() is True and view._footprint_merged is True
        window.footprint_by_owner_action.setChecked(True)
        assert settings.get_footprint_by_owner() is True and view._footprint_by_owner is True
        assert len(view._footprint_items) > 1
        window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_UNITS].setChecked(True)
        assert settings.get_footprint_scope() == unit_pick.FOOTPRINT_SCOPE_UNITS
        assert view._footprint_scope == unit_pick.FOOTPRINT_SCOPE_UNITS
        window.footprint_action.setChecked(False)
        assert not any(item.isVisible() for item in view._footprint_items.values())
        window.footprint_action.setChecked(True)
        assert all(item.isVisible() for item in view._footprint_items.values())
        window.footprint_action.setChecked(False)
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            for action in (window.footprint_action, window.footprint_merged_action, window.footprint_by_owner_action):
                assert action.isEnabled(), (style, action.text())
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_window_reads_the_new_toggles_from_config_without_writing_it(tmp_path) -> None:
    """A fresh window ticks Units only, Merge and Colour by Owner from the file and writes nothing."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    config = tmp_path / "config.yaml"
    config.write_text("footprint_scope: units\nfootprint_merged: true\nfootprint_by_owner: true\n")
    before = config.read_bytes()
    window = ViewerWindow()
    try:
        assert window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_UNITS].isChecked()
        assert window.footprint_merged_action.isChecked()
        assert window.footprint_by_owner_action.isChecked()
        assert window.map_view._footprint_merged is True
        assert window.map_view._footprint_by_owner is True
        assert config.read_bytes() == before
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
        view = window.map_view
        view.set_footprint_by_owner(True)
        assert len(view._footprint_items) > 1, "no owner group, so this proves nothing"
        assert {item.zValue() for item in view._footprint_items.values()} == {MapView.FOOTPRINT_Z}
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_closing_a_map_then_toggling_does_not_touch_a_deleted_item() -> None:
    window = _window_with_units("Stepped")
    try:
        view = window.map_view
        view.set_footprint_by_owner(True)
        assert len(view._footprint_items) > 1, "no owner group to tear down, so this proves nothing"
        view.clear_image()
        assert view._footprint_items == {}
        assert _outline_items(view) == []
        view.set_footprint_outlines(True)
        view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_ALL)
        view.set_footprint_merged(True)
        view.set_footprint_by_owner(False)
        view.schedule_footprint_refresh()
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
        before = _overlay_item(view).path().elementCount()
        placed_before = len(window.scenario.unit_manager.units[1])
        window.units_panel.select_object(_TOWN_CENTRE)
        # on_unit_place takes a SCENE point (it calls _pick_tile directly).
        window.on_unit_place(view._tile_polygon(50, 50).boundingRect().center(), None)
        assert len(window.scenario.unit_manager.units[1]) == placed_before + 1, "the place did not happen"
        assert _overlay_item(view).path().elementCount() > before
    finally:
        conftest.close_window(window)


def _outline_items(view):
    """The overlay found by its z layer, not through the private item ref."""
    from descape.map_view import MapView

    return [item for item in view.scene().items() if item.zValue() == MapView.FOOTPRINT_Z]


def _outline_points(view) -> set[tuple[float, float]]:
    (item,) = _outline_items(view)
    path = item.path()
    return {(round(path.elementAt(i).x, 3), round(path.elementAt(i).y, 3)) for i in range(path.elementCount())}


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_colour_change_recolours_the_outline_pen() -> None:
    """GH #86 colour step, per test_ruler_viewer.py's recoloured-glow test."""
    window = _window_with_units("Stepped")
    try:
        view = window.map_view
        items = _outline_items(view)
        assert len(items) == 1
        assert items[0].pen().color().name() != "#123456"
        settings.set_overlay_color("footprint_outline", "#123456")
        view.apply_overlay_colors()
        assert _outline_items(view)[0].pen().color().name() == "#123456"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_moving_a_unit_moves_its_outline_and_undo_puts_it_back() -> None:
    """GH #86 move + undo, on a real placed Town Centre (UnitEditModel edits
    the real scenario objects, not this module's duck-typed ones)."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        view = window.map_view
        window.footprint_action.setChecked(True)
        view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_ALL)
        window.mode_combo.setCurrentText("Units")
        window.show()
        QApplication.processEvents()
        window.units_panel.select_object(_TOWN_CENTRE)
        window.on_unit_place(view._tile_polygon(50, 50).boundingRect().center(), None)
        (entry,) = [e for e in view._unit_index.entries if e.unit.unit_const == _TOWN_CENTRE]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        view.set_unit_selection([entry])
        placed = _outline_points(view)
        assert placed, "the placed Town Centre drew no outline"
        start_x = entry.unit.x

        window.on_unit_nudge(1, 0, Qt.ShiftModifier)
        assert view._unit_index.entry_for_key(key).unit.x == pytest.approx(start_x + 1.0), "the nudge did not happen"
        moved = _outline_points(view)
        assert moved != placed, "the move left the outline where it was"

        window.undo()
        assert _outline_points(view) == placed
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
        before = _overlay_item(view).path().elementCount()
        assert window.mode != "units"
        units = window.scenario.unit_manager.units
        units[1].append(_Unit(50.0, 50.0, _TOWN_CENTRE, reference_id=301))
        window._after_unit_mutation()
        assert _overlay_item(view).path().elementCount() > before
    finally:
        conftest.close_window(window)


# --- outlines off: nothing built while hidden, rebuilt on enable ------------


def _expected_points(view) -> set[tuple[float, float]]:
    """Freshly recomputed from the live index and heights, the hover cue's geometry."""
    points = set()
    for entry in unit_pick.footprint_entries(view._unit_index, view._footprint_scope):
        for polygon in view._unit_polygons_for(entry) or []:
            points |= {(round(x, 3), round(y, 3)) for x, y in polygon}
    return points


def _drawn_points(view) -> set[tuple[float, float]]:
    path = _overlay_item(view).path()
    return {(round(path.elementAt(i).x, 3), round(path.elementAt(i).y, 3)) for i in range(path.elementCount())}


def _units_window_outlines_off():
    """Blank template in Units mode with outlines left at their default, off."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    assert settings.get_footprint_outlines() is False
    window.mode_combo.setCurrentText("Units")
    window.show()
    QApplication.processEvents()
    assert window.mode == "units"
    return window


def _place(window, const: int, tile: tuple[int, int]) -> None:
    placed_before = sum(len(units) for units in window.scenario.unit_manager.units)
    window.units_panel.select_object(const)
    window.on_unit_place(window.map_view._tile_polygon(*tile).boundingRect().center(), None)
    assert sum(len(units) for units in window.scenario.unit_manager.units) == placed_before + 1, "no place"


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_unit_placed_with_outlines_off_is_outlined_once_they_are_enabled() -> None:
    """The hidden path is not built, and enabling rebuilds from the live index."""
    window = _units_window_outlines_off()
    try:
        view = window.map_view
        _place(window, _TOWN_CENTRE, (50, 50))
        assert _overlay_item(view).path().elementCount() == 0, "the hidden path was built"
        window.footprint_action.setChecked(True)
        (entry,) = [e for e in view._unit_index.entries if e.unit.unit_const == _TOWN_CENTRE]
        placed = {(round(x, 3), round(y, 3)) for polygon in view._unit_polygons_for(entry) for x, y in polygon}
        drawn = _drawn_points(view)
        assert placed and placed <= drawn
        assert drawn == _expected_points(view)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_an_elevation_edit_with_outlines_off_is_outlined_at_the_new_height_once_enabled() -> None:
    """No refresh is queued per touch while hidden; enabling draws the new heights."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _units_window_outlines_off()
    try:
        view = window.map_view
        _place(window, _TOWN_CENTRE, (20, 20))
        at_old_height = _expected_points(view)
        window.mode_combo.setCurrentText("Terrain")
        assert view._unit_index is not None
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(20, 20, Qt.NoModifier)
        assert not view._footprint_refresh_pending, "a hidden overlay queued a rebuild"
        window.on_edit_stroke_end()
        QApplication.processEvents()
        window.footprint_action.setChecked(True)
        drawn = _drawn_points(view)
        assert drawn == _expected_points(view)
        assert drawn != at_old_height, "the edit did not move the outline"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_unit_edit_with_outlines_off_builds_no_outline_geometry(monkeypatch) -> None:
    """A Move rather than a Place: Place selects the placed unit, whose cue is
    also unit_polygons(). Both reach refresh_after_index_patch() alike."""
    window = _units_window_outlines_off()
    try:
        view = window.map_view
        _place(window, _TOWN_CENTRE, (50, 50))
        window._selection = []
        view.set_unit_selection([])
        (entry,) = [e for e in view._unit_index.entries if e.unit.unit_const == _TOWN_CENTRE]
        start_x = entry.unit.x
        calls = []
        real = unit_pick.unit_polygons
        monkeypatch.setattr(unit_pick, "unit_polygons", lambda *a, **k: calls.append(1) or real(*a, **k))
        window._move_units(window._ensure_unit_edits(), [entry], 1.0, 0.0, "Move unit")
        assert entry.unit.x == pytest.approx(start_x + 1.0), "the move did not happen"
        assert calls == []
    finally:
        conftest.close_window(window)


def _count_footprint_builds(monkeypatch, view) -> list[int]:
    calls: list[int] = []
    real = view._unit_polygons_for
    monkeypatch.setattr(view, "_unit_polygons_for", lambda entry: calls.append(1) or real(entry))
    return calls


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_an_elevation_stroke_with_outlines_off_builds_no_outline_geometry_through_its_release(monkeypatch) -> None:
    """set-elevation-untimed-stalls plan 4a: nothing is built by the whole stroke,
    release and event-loop turn included, not only nothing queued mid-stroke."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _units_window_outlines_off()
    try:
        view = window.map_view
        _place(window, _TOWN_CENTRE, (20, 20))
        window.mode_combo.setCurrentText("Terrain")
        assert view._unit_index is not None, "no live index -- vacuous"
        before = window.scenario.map_manager.get_tile(20, 20).elevation
        calls = _count_footprint_builds(monkeypatch, view)
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(20, 20, Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        assert window.scenario.map_manager.get_tile(20, 20).elevation != before, "the stroke wrote nothing"
        assert calls == []
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_scope_change_with_outlines_off_builds_nothing_and_is_drawn_once_enabled(monkeypatch) -> None:
    """set-elevation-untimed-stalls plan 4a: the hidden overlay skips the scope's
    rebuild, and enabling draws the new scope (the villager only `all` outlines)."""
    from PyQt5.QtWidgets import QApplication

    window = _units_window_outlines_off()
    try:
        view = window.map_view
        _place(window, _TOWN_CENTRE, (50, 50))
        _place(window, _VILLAGER, (30, 30))
        assert view._footprint_scope == unit_pick.FOOTPRINT_SCOPE_MULTITILE
        calls = _count_footprint_builds(monkeypatch, view)
        window._on_footprint_scope(unit_pick.FOOTPRINT_SCOPE_ALL)
        QApplication.processEvents()
        assert calls == [], "the hidden overlay was rebuilt for the new scope"
        window.footprint_action.setChecked(True)
        (villager,) = [e for e in view._unit_index.entries if e.unit.unit_const == _VILLAGER]
        villager_points = {(round(x, 3), round(y, 3)) for p in view._unit_polygons_for(villager) for x, y in p}
        drawn = _drawn_points(view)
        assert villager_points and villager_points <= drawn, "enabling drew the old scope"
        assert drawn == _expected_points(view)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_place_that_stacks_adds_a_badge_and_a_move_that_stacks_nothing_skips_the_rebuild() -> None:
    window = _units_window_outlines_off()
    view = window.map_view
    try:
        assert settings.get_stack_badges() is True
        _place(window, _VILLAGER, (40, 40))
        _place(window, _VILLAGER, (40, 40))
        assert len(view._stack_groups) == 1, "the second villager did not stack"
        assert view._stack_badge_item.badge_texts() == ["2"]
        _place(window, _TOWN_CENTRE, (60, 60))
        window._selection = []
        view.set_unit_selection([])
        (entry,) = [e for e in view._unit_index.entries if e.unit.unit_const == _TOWN_CENTRE]
        start_x = entry.unit.x
        rebuilds = []
        real = view._rebuild_stack_badges
        view._rebuild_stack_badges = lambda: (rebuilds.append(1), real())[1]
        window._move_units(window._ensure_unit_edits(), [entry], 1.0, 0.0, "Move unit")
        assert entry.unit.x == pytest.approx(start_x + 1.0), "the move did not happen"
        assert rebuilds == []
        assert view._stack_badge_item.badge_texts() == ["2"]
    finally:
        view.__dict__.pop("_rebuild_stack_badges", None)
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
            path = _overlay_item(view).path()
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


# --- GH #143: Colour by Owner ------------------------------------------------


def _owner_window():
    """_window_with_units' P1 buildings plus a P2 house and a GAIA tree, at All units."""
    window = _window_with_units("Stepped")
    units = window.scenario.unit_manager.units
    units[2].append(_Unit(50.0, 50.0, _HOUSE, reference_id=501))
    units[0].append(_Unit(60.5, 60.5, _TREE, reference_id=502))
    window._after_unit_mutation()
    window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_ALL].setChecked(True)
    return window


def _group_points(item) -> set[tuple[float, float]]:
    path = item.path()
    return {(round(path.elementAt(i).x, 3), round(path.elementAt(i).y, 3)) for i in range(path.elementCount())}


def _entry_points(view, player_id: int) -> set[tuple[float, float]]:
    return {
        (round(x, 3), round(y, 3))
        for entry in unit_pick.footprint_entries(view._unit_index, view._footprint_scope)
        if entry.player_id == player_id
        for polygon in view._unit_polygons_for(entry) or []
        for x, y in polygon
    }


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_colour_by_owner_puts_each_player_in_their_colour_and_gaia_in_the_configured_one() -> None:
    window = _owner_window()
    try:
        view = window.map_view
        colors = window.scenario.player_colors
        p1, p2 = tuple(colors[1]), tuple(colors[2])
        assert p1 != p2
        assert list(view._footprint_items) == [None], "Colour by Owner ships off"
        window.footprint_by_owner_action.setChecked(True)
        assert set(view._footprint_items) == {None, p1, p2}
        for key in (p1, p2):
            pen = view._footprint_items[key].pen()
            assert pen.color().getRgb()[:3] == key
            assert pen.widthF() == 0
        assert view._footprint_items[None].pen().color() == view._footprint_pen.color()
        assert _group_points(view._footprint_items[None]) == _entry_points(view, 0)
        assert _group_points(view._footprint_items[p1]) == _entry_points(view, 1)
        assert _group_points(view._footprint_items[p2]) == _entry_points(view, 2)

        window.footprint_by_owner_action.setChecked(False)
        assert list(view._footprint_items) == [None]
        assert _outline_items(view) == [view._footprint_items[None]]
        assert _drawn_points(view) == _expected_points(view)
    finally:
        conftest.close_window(window)


def _change_player_colour(window, pid: int) -> tuple[int, int, int]:
    from descape import player_fields

    spec = {s.field_id: s for s in player_fields.specs_for(window.scenario)}["color"]
    # Off both owners' current colours, so the two groups never share a key.
    taken = {player_fields.current_value(window.scenario, spec, p) for p in (1, 2)}
    new_value = next(value for value, _label in spec.choices if value not in taken)
    old_rgb = tuple(window.scenario.player_colors[pid])
    window.set_player_field(spec, pid, new_value)
    new_rgb = tuple(window.scenario.player_colors[pid])
    assert new_rgb != old_rgb, "the colour edit changed nothing"
    return new_rgb


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_player_colour_edit_recolours_the_owner_outlines() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _owner_window()
    try:
        view = window.map_view
        window.footprint_by_owner_action.setChecked(True)
        old_p2 = tuple(window.scenario.player_colors[2])
        p2_points = _group_points(view._footprint_items[old_p2])
        new_p2 = _change_player_colour(window, 2)
        QApplication.processEvents()
        assert old_p2 not in view._footprint_items or old_p2 == tuple(window.scenario.player_colors[1])
        assert view._footprint_items[new_p2].pen().color().getRgb()[:3] == new_p2
        assert _group_points(view._footprint_items[new_p2]) == p2_points
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_one_index_rebuild_builds_the_owner_coloured_overlay_once(monkeypatch) -> None:
    """_rebuild_unit_index pushes colours, then set_unit_index refreshes synchronously:
    a colour-push hook would queue a second build, which processEvents() would run."""
    from PyQt5.QtWidgets import QApplication

    window = _owner_window()
    try:
        view = window.map_view
        window.footprint_by_owner_action.setChecked(True)
        QApplication.processEvents()
        in_scope = len(unit_pick.footprint_entries(view._unit_index, view._footprint_scope))
        assert in_scope >= 6
        calls = _count_footprint_builds(monkeypatch, view)
        window._rebuild_unit_index()
        QApplication.processEvents()
        assert len(calls) == in_scope
        calls.clear()
        _change_player_colour(window, 2)
        QApplication.processEvents()
        assert len(calls) == in_scope
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_player_colour_edit_builds_nothing_while_outlines_use_one_colour(monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    window = _owner_window()
    try:
        view = window.map_view
        QApplication.processEvents()
        calls = _count_footprint_builds(monkeypatch, view)
        _change_player_colour(window, 2)
        QApplication.processEvents()
        assert calls == []
    finally:
        conftest.close_window(window)
