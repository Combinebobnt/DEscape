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
        # The 4x4 snaps to a whole number (GH #206), so its own tile is int(x), int(y), not the clicked one.
        (centre,) = [e.unit for e in view._unit_index.entries if e.unit.unit_const == _TOWN_CENTRE]
        own = (int(centre.x), int(centre.y))
        at_old_height = _expected_points(view)
        # Place selected it, and Terrain keeps a selection whose highlight queues its own refresh (TASK-359).
        window.deselect_units()
        window.mode_combo.setCurrentText("Terrain")
        assert view._unit_index is not None
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(*own, Qt.NoModifier)
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
        # Place selected it, and a selection kept in Terrain draws through unit_polygons() too (TASK-359).
        window.deselect_units()
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
        window.unrestricted_check.setChecked(True)  # the second villager goes on the first
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


def _team_colour(window, player_id: int) -> tuple[int, int, int]:
    """The owner's sprite tint, which an owner group is keyed and inked by (TASK-449)."""
    from descape.unit_sprites import TEAM_COLORS

    return tuple(TEAM_COLORS[window.scenario.team_indices[player_id] % len(TEAM_COLORS)])


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
        p1, p2 = _team_colour(window, 1), _team_colour(window, 2)
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
    old_rgb = _team_colour(window, pid)
    window.set_player_field(spec, pid, new_value)
    new_rgb = _team_colour(window, pid)
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
        old_p2 = _team_colour(window, 2)
        p2_points = _group_points(view._footprint_items[old_p2])
        new_p2 = _change_player_colour(window, 2)
        QApplication.processEvents()
        assert old_p2 not in view._footprint_items or old_p2 == _team_colour(window, 1)
        assert view._footprint_items[new_p2].pen().color().getRgb()[:3] == new_p2
        assert _group_points(view._footprint_items[new_p2]) == p2_points
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_one_index_rebuild_builds_the_owner_coloured_overlay_once(monkeypatch) -> None:
    """_rebuild_unit_index pushes colours, then set_unit_index refreshes synchronously:
    a colour-push hook would queue a second refresh, which processEvents() would run.
    An unchanged index rebuilds no chunk (TASK-031.42), so refreshes are counted there."""
    from PyQt5.QtWidgets import QApplication

    window = _owner_window()
    try:
        view = window.map_view
        window.footprint_by_owner_action.setChecked(True)
        QApplication.processEvents()
        in_scope = len(unit_pick.footprint_entries(view._unit_index, view._footprint_scope))
        assert in_scope >= 6
        calls = _count_footprint_builds(monkeypatch, view)
        refreshes = []
        real_refresh = view.refresh_footprint_overlay
        monkeypatch.setattr(view, "refresh_footprint_overlay", lambda: refreshes.append(1) or real_refresh())
        window._rebuild_unit_index()
        QApplication.processEvents()
        assert len(refreshes) == 1
        assert calls == []
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


# --- the owed in-app passes, offscreen (TASK-131) ------------------------------

_DOS_PILAS = "F7_2_Dos Pilas"


def _dos_pilas_window(corpus_files, style: str = "Stepped"):
    """F7_2_Dos Pilas (648) with View > Footprint Outlines > Show on, in `style`."""
    from PyQt5.QtWidgets import QApplication

    path = next((p for p in corpus_files if _DOS_PILAS in p.name), None)
    if path is None:
        pytest.skip(f"{_DOS_PILAS} is not in this run's corpus_files")
    window = conftest.stepped_window(path)
    window.terrain_style_combo.setCurrentText(style)
    window.refresh_map()
    QApplication.processEvents()
    if style == "Flat":
        # Offscreen, a restyle never reaches MapView (GOTCHAS); safe for geometry-only checks.
        window.map_view._terrain_style = "flat"
    window.footprint_action.setChecked(True)
    return window


def _points_of(view, entries, merged: bool = False) -> set[tuple[float, float]]:
    points = set()
    for entry in entries:
        polygons = view._unit_polygons_for(entry) or []
        if merged:
            polygons = unit_pick.merge_footprint_polygons(polygons)
        points |= {(round(x, 3), round(y, 3)) for polygon in polygons for x, y in polygon}
    return points


@pytest.mark.corpus
@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_units_only_on_dos_pilas_drops_every_building_wall_and_gate_and_keeps_every_unit(corpus_files) -> None:
    """GH #143 step 1 on the real file, through View > Footprint Outlines > Units only.

    Pass conditions:
    - the file has buildings/walls/gates and other units, so the scope has something to cut;
    - every index entry whose const is a building, wall or gate (Show Buildings/Show Walls sets)
      is out of scope, unless it is mobile siege; every other entry is in scope;
    - the drawn outline is exactly the in-scope entries' polygons, and has fewer points than All
      units draws.
    (Dos Pilas has no trebuchet; Units only keeping siege is the synthetic
    test_units_only_drops_buildings_walls_and_gates_and_keeps_siege.)
    """
    from descape import unit_kind

    window = _dos_pilas_window(corpus_files)
    try:
        view = window.map_view
        window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_UNITS].setChecked(True)
        assert view._footprint_scope == unit_pick.FOOTPRINT_SCOPE_UNITS
        structures = unit_kind.building_consts() | unit_kind.wall_consts()
        entries = view._unit_index.entries
        in_scope = {id(e) for e in unit_pick.footprint_entries(view._unit_index, view._footprint_scope)}
        dropped = [e for e in entries if e.unit.unit_const in structures and e.unit.unit_const not in _SIEGE_IN_BOTH_SCOPES]
        kept = [e for e in entries if e.unit.unit_const not in structures]
        assert dropped and kept
        assert not any(id(e) in in_scope for e in dropped)
        assert all(id(e) in in_scope for e in kept)
        drawn = _drawn_points(view)
        assert drawn == _points_of(view, [e for e in entries if id(e) in in_scope])
        window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_ALL].setChecked(True)
        assert len(drawn) < len(_drawn_points(view))
    finally:
        conftest.close_window(window)


@pytest.mark.corpus
@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_merge_on_dos_pilas_outlines_each_town_centre_once(corpus_files, style: str) -> None:
    """GH #143 step 2's town-centre half on the real file, Multi-tile objects scope.

    Pass conditions, with Merge ticked through its menu action:
    - every Town Centre (109) on the file merges to exactly one ring (16 per-tile polygons in
      Stepped and Sloped, one rect in Flat);
    - the drawn outline is exactly the in-scope entries' merged rings, with one subpath per
      ring, and in Stepped and Sloped fewer subpaths than with Merge off.
    """
    window = _dos_pilas_window(corpus_files, style)
    try:
        view = window.map_view
        assert view._terrain_style == style.lower()
        window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_MULTITILE].setChecked(True)
        per_tile = _subpath_count(_overlay_item(view).path())
        window.footprint_merged_action.setChecked(True)
        entries = unit_pick.footprint_entries(view._unit_index, view._footprint_scope)
        centres = [e for e in entries if e.unit.unit_const == _TOWN_CENTRE]
        assert centres
        for entry in centres:
            polygons = view._unit_polygons_for(entry)
            assert len(polygons) == (1 if style == "Flat" else 16)
            assert len(unit_pick.merge_footprint_polygons(polygons)) == 1
        rings = sum(len(unit_pick.merge_footprint_polygons(view._unit_polygons_for(e) or [])) for e in entries)
        merged = _overlay_item(view).path()
        assert _subpath_count(merged) == rings
        assert _drawn_points(view) == _points_of(view, entries, merged=True)
        if style != "Flat":
            assert rings < per_tile
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_merge_in_sloped_with_sprites_joins_a_gently_draped_farm_and_leaves_a_multi_level_one_per_tile() -> None:
    """GH #143 step 2's farm half (Dos Pilas has no farm), Sloped with Show sprites on.

    A farm on a one-level-per-tile ramp, and one straddling a sharp 0 -> 4 step in y, whose
    draped staircase leaves several loops. Pass conditions, with Merge ticked:
    - both farms are draped 3x3 footprints (nine per-tile polygons each);
    - the drawn outline is one ring for the gentle farm plus the steep farm's nine per-tile
      polygons, ten subpaths in all, and those points exactly.
    """
    from PyQt5.QtWidgets import QApplication

    gentle, steep = (21.5, 21.5), (60.5, 46.5)
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        for tile in window.scenario.map_manager.terrain:
            if tile.y < 35:
                tile.elevation = max(0, min(4, tile.x - 18))
            else:
                tile.elevation = 4 if tile.y >= 46 else 0
        units = window.scenario.unit_manager.units[1]
        units.append(_Unit(*gentle, _FARM, reference_id=701))
        units.append(_Unit(*steep, _FARM, reference_id=702))
        window.terrain_style_combo.setCurrentText("Sloped")
        window.refresh_map()
        QApplication.processEvents()
        assert window.show_sprites_action.isChecked()
        window.footprint_action.setChecked(True)
        view = window.map_view
        window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_BUILDINGS].setChecked(True)
        window.footprint_merged_action.setChecked(True)
        farms = {(e.unit.x, e.unit.y): e for e in view._unit_index.entries if e.unit.unit_const == _FARM}
        assert set(farms) == {gentle, steep}
        for entry in farms.values():
            assert len(view._unit_polygons_for(entry)) == 9
        assert len(unit_pick.merge_footprint_polygons(view._unit_polygons_for(farms[gentle]))) == 1
        assert len(unit_pick.merge_footprint_polygons(view._unit_polygons_for(farms[steep]))) == 9
        assert _subpath_count(_overlay_item(view).path()) == 1 + 9
        assert _drawn_points(view) == _points_of(view, farms.values(), merged=True)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_scope_merge_and_owner_colour_ticked_in_the_menu_survive_a_restart(monkeypatch) -> None:
    """GH #143 step 5, end to end. Pass conditions: after Units only, Merge tiles and Colour by
    Owner are ticked in the View menu and the window closes, a restart (settings memos dropped,
    a new window) shows all three ticked and MapView using all three."""
    from descape.viewer import ViewerWindow

    window = _window_with_units("Stepped")
    try:
        window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_UNITS].setChecked(True)
        window.footprint_merged_action.setChecked(True)
        window.footprint_by_owner_action.setChecked(True)
    finally:
        conftest.close_window(window)
    for name in ("_footprint_outlines", "_footprint_scope", "_footprint_merged", "_footprint_by_owner"):
        monkeypatch.setattr(settings, name, None)
    window = ViewerWindow()
    try:
        assert window.footprint_action.isChecked()
        assert window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_UNITS].isChecked()
        assert window.footprint_merged_action.isChecked()
        assert window.footprint_by_owner_action.isChecked()
        view = window.map_view
        assert view._footprint_scope == unit_pick.FOOTPRINT_SCOPE_UNITS
        assert view._footprint_merged is True and view._footprint_by_owner is True
    finally:
        conftest.close_window(window)


# --- View > Layers > Fish Outlines (TASK-499) --------------------------------

_SHORE_FISH, _BOX_TURTLES, _OYSTERS, _DORADO, _WHALE = 69, 1141, 2170, 455, 2625
_TURTLE_SHIP, _GOLD_MINE = 831, 66
# Each fish's tile count: three 1x1s, a 2x2 deep fish and the 3x3 Whale.
_FISH_TILES = {_SHORE_FISH: 1, _BOX_TURTLES: 1, _OYSTERS: 1, _DORADO: 4, _WHALE: 9}


def _window_with_fish(style: str = "Stepped"):
    """GAIA fish of every size plus a Turtle Ship and a gold mine, the two
    near misses, with Fish Outlines on and Footprint Outlines left off."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    units = window.scenario.unit_manager.units
    units[0].append(_Unit(10.5, 10.5, _SHORE_FISH, reference_id=401))
    units[0].append(_Unit(20.5, 10.5, _BOX_TURTLES, reference_id=402))
    units[0].append(_Unit(30.5, 10.5, _OYSTERS, reference_id=403))
    units[0].append(_Unit(40.0, 10.0, _DORADO, reference_id=404))
    units[0].append(_Unit(50.5, 10.5, _WHALE, reference_id=405))
    units[0].append(_Unit(60.5, 10.5, _GOLD_MINE, reference_id=406))
    units[1].append(_Unit(70.5, 10.5, _TURTLE_SHIP, reference_id=407))
    window.terrain_style_combo.setCurrentText(style)
    window.refresh_map()
    QApplication.processEvents()
    window.layer_actions["fish_outlines"].setChecked(True)
    view = window.map_view
    if style == "Flat":
        # Offscreen, a restyle never reaches MapView (GOTCHAS); safe for geometry-only checks.
        view._terrain_style = "flat"
        view.refresh_fish_overlay()
    assert view._terrain_style == style.lower()
    return window


def _fish_rings(view) -> list[list[tuple[float, float]]]:
    from PyQt5.QtGui import QPainterPath

    path = view._fish_item.path()
    rings: list[list[tuple[float, float]]] = []
    for i in range(path.elementCount()):
        element = path.elementAt(i)
        if element.type == QPainterPath.MoveToElement:
            rings.append([])
        rings[-1].append((element.x, element.y))
    return rings


def _area(points) -> float:
    pairs = zip(points, [*points[1:], points[0]], strict=True)
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in pairs)) / 2


def _fish_points(view) -> set[tuple[float, float]]:
    return {(round(x, 3), round(y, 3)) for ring in _fish_rings(view) for x, y in ring}


def _unit_points(view, entry) -> set[tuple[float, float]]:
    return {(round(x, 3), round(y, 3)) for polygon in view._unit_polygons_for(entry) for x, y in polygon}


def _expected_fish_points(view) -> set[tuple[float, float]]:
    """Freshly recomputed: every indexed fish's merged ring, from the live heights."""
    points = set()
    for entry in view._unit_index.entries:
        if entry.unit.unit_const in _FISH_TILES:
            for ring in unit_pick.merge_footprint_polygons(view._unit_polygons_for(entry)):
                points |= {(round(x, 3), round(y, 3)) for x, y in ring}
    return points


@pytest.mark.parametrize("style", ["Flat", "Stepped"])
@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_fish_outlines_draw_one_ring_per_fish_of_its_whole_footprint(style: str) -> None:
    """Five fish, five rings, of 1, 1, 1, 4 and 9 tiles' area, with Footprint
    Outlines off; the Turtle Ship (a warship) and the gold mine get none."""
    window = _window_with_fish(style)
    try:
        view = window.map_view
        assert not window.footprint_action.isChecked()
        tile = [(p.x(), p.y()) for p in view._tile_polygon(5, 5)]
        areas = sorted(round(_area(ring) / _area(tile), 3) for ring in _fish_rings(view))
        assert areas == sorted(_FISH_TILES.values())
        by_const = {e.unit.unit_const: e for e in view._unit_index.entries}
        drawn = _fish_points(view)
        assert drawn == _expected_fish_points(view)
        for const in _FISH_TILES:
            assert _unit_points(view, by_const[const]) & drawn, const
        for const in (_TURTLE_SHIP, _GOLD_MINE):
            assert not _unit_points(view, by_const[const]) & drawn, const
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_fish_outlines_ignore_footprint_scope_merge_and_colour_and_sit_above_them() -> None:
    from descape.map_view import MapView

    window = _window_with_fish("Stepped")
    try:
        view = window.map_view
        before = _fish_points(view)
        window.footprint_action.setChecked(True)
        window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_BUILDINGS].setChecked(True)
        window.footprint_merged_action.setChecked(False)
        window.footprint_by_owner_action.setChecked(True)
        assert _fish_points(view) == before
        assert len(_fish_rings(view)) == len(_FISH_TILES)
        assert MapView.FOOTPRINT_Z < view._fish_item.zValue() < MapView.RANGE_RING_Z
        pen = view._fish_item.pen()
        assert pen.isCosmetic() and pen.width() == MapView.FISH_OUTLINE_PEN_WIDTH
        assert pen.color().name() == settings.get_default_overlay_color("fish_outline")
        settings.set_overlay_color("fish_outline", "#123456")
        view.apply_overlay_colors()
        assert view._fish_item.pen().color().name() == "#123456"
        assert view._fish_item.pen().width() == MapView.FISH_OUTLINE_PEN_WIDTH
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_fish_outlines_follow_the_unit_filter() -> None:
    """Every fish is GAIA's, so Filters > Show GAIA unticked leaves no ring."""
    window = _window_with_fish("Stepped")
    try:
        view = window.map_view
        assert len(_fish_rings(view)) == len(_FISH_TILES)
        window.show_gaia_action.setChecked(False)
        assert _fish_rings(view) == []
        window.show_gaia_action.setChecked(True)
        assert len(_fish_rings(view)) == len(_FISH_TILES)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_moving_a_fish_moves_its_ring_and_undo_puts_it_back() -> None:
    """A move patches the index in place (refresh_after_index_patch), the path
    a whole-index refresh would not cover. Free placement lifts the water rule."""
    window = _units_window_outlines_off()
    try:
        view = window.map_view
        window.layer_actions["fish_outlines"].setChecked(True)
        window.unrestricted_check.setChecked(True)
        # pick_object(), not _place(): the catalog lists a GAIA-only Whale for GAIA alone (GH #32).
        assert window.units_panel.pick_object(_WHALE)
        window.on_unit_place(view._tile_polygon(50, 50).boundingRect().center(), None)
        (entry,) = [e for e in view._unit_index.entries if e.unit.unit_const == _WHALE]
        placed = _fish_points(view)
        assert placed and placed <= _unit_points(view, entry)
        start_x = entry.unit.x
        window._move_units(window._ensure_unit_edits(), [entry], 1.0, 0.0, "Move unit")
        assert entry.unit.x == pytest.approx(start_x + 1.0), "the move did not happen"
        moved = _fish_points(view)
        assert moved != placed, "the move left the ring where it was"
        assert moved == _expected_fish_points(view)
        window.undo()
        assert _fish_points(view) == placed
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_an_elevation_edit_resyncs_the_fish_rings_with_footprints_off() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _window_with_fish("Stepped")
    try:
        view = window.map_view
        before = _fish_points(view)
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(50, 10, Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        after = _fish_points(view)
        assert after != before, "the edit did not move any ring"
        assert after == _expected_fish_points(view)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_closing_a_map_then_toggling_fish_outlines_does_not_touch_a_deleted_item() -> None:
    window = _window_with_fish("Stepped")
    try:
        view = window.map_view
        view.clear_image()
        assert view._fish_item is None
        view.set_fish_outlines(False)
        view.set_fish_outlines(True)
        view.apply_overlay_colors()
        view.schedule_footprint_refresh()
    finally:
        conftest.close_window(window)


# --- TASK-031.42: only the chunks an edit touched are rebuilt -----------------

# outline_chunks.CHUNK_TILES in these tests, so a small fixture spans several chunks.
_CHUNK = 4
_GATE_NE = 64


def _subpaths(path) -> list[tuple]:
    """The path's subpaths as a sorted multiset of point tuples: a stale or dropped copy counts."""
    from PyQt5.QtGui import QPainterPath

    subpaths: list[list] = []
    for i in range(path.elementCount()):
        element = path.elementAt(i)
        if element.type == QPainterPath.MoveToElement:
            subpaths.append([])
        subpaths[-1].append((round(element.x, 3), round(element.y, 3)))
    return sorted(tuple(points) for points in subpaths)


def _fresh_groups(view) -> dict:
    """Each owner group's subpaths built from scratch, as the unchunked overlay did."""
    from PyQt5.QtGui import QPainterPath

    from descape.map_view import _add_closed_polygons

    groups = {None: QPainterPath()}
    for entry in unit_pick.footprint_entries(view._unit_index, view._footprint_scope):
        polygons = view._unit_polygons_for(entry)
        if not polygons:
            continue
        if view._footprint_merged:
            polygons = unit_pick.merge_footprint_polygons(polygons)
        key = view._owner_group_key(entry.player_id, view._footprint_by_owner)
        _add_closed_polygons(groups.setdefault(key, QPainterPath()), polygons)
    return {key: _subpaths(path) for key, path in groups.items()}


def _assert_fresh(view, step: str) -> None:
    fresh = _fresh_groups(view)
    drawn = {key: _subpaths(item.path()) for key, item in view._footprint_items.items()}
    assert set(drawn) == set(fresh), f"{step}: groups"
    for key, subpaths in fresh.items():
        assert drawn[key] == subpaths, f"{step}: group {key} differs from a fresh build"


def _chunk_elements(view) -> dict:
    """Every item's chunks as ordered path elements: subpath order changes stroked pixels at low zoom."""
    return {
        key: {
            chunk: [(e.type, e.x, e.y) for e in (path.elementAt(i) for i in range(path.elementCount()))]
            for chunk, path in item.chunks().items()
        }
        for key, item in view._footprint_items.items()
    }


def _assert_same_as_a_full_rebuild(view, step: str) -> None:
    """Each chunk's path, in order, equals what a full rebuild writes (TASK-031.42 M check)."""
    incremental = _chunk_elements(view)
    view._footprint_chunks.reset()
    view.refresh_footprint_overlay()
    assert incremental == _chunk_elements(view), f"{step}: a chunk's path differs from a full rebuild's"


def _chunked_window(monkeypatch, style: str = "Stepped", merged: bool = False, by_owner: bool = False):
    """Blank template, Units mode, outlines on at scope All, free placement, small chunks."""
    from PyQt5.QtWidgets import QApplication

    from descape import outline_chunks

    monkeypatch.setattr(outline_chunks, "CHUNK_TILES", _CHUNK)
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    if style != "Stepped":
        window.terrain_style_combo.setCurrentText(style)
        window.refresh_map()
        QApplication.processEvents()
    view = window.map_view
    if style == "Flat":
        # Offscreen, a restyle never reaches MapView (GOTCHAS); safe for geometry-only checks.
        view._terrain_style = "flat"
    assert view._terrain_style == style.lower()
    window.footprint_action.setChecked(True)
    window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_ALL].setChecked(True)
    window.footprint_merged_action.setChecked(merged)
    window.footprint_by_owner_action.setChecked(by_owner)
    window.unrestricted_check.setChecked(True)
    window.mode_combo.setCurrentText("Units")
    window.show()
    QApplication.processEvents()
    return window


def _keys(window) -> set:
    return {(pid, u.reference_id) for pid, units in enumerate(window.scenario.unit_manager.units) for u in units}


def _place_for(window, owner: int, const: int, tile: tuple[int, int]) -> tuple[int, int]:
    from PyQt5.QtWidgets import QApplication

    before = _keys(window)
    window.units_panel.select_owner(owner)
    assert window.units_panel.pick_object(const), const
    window.on_unit_place(window.map_view._tile_polygon(*tile).boundingRect().center(), None)
    QApplication.processEvents()
    (key,) = _keys(window) - before
    return key


def _deselect(window) -> None:
    window._selection = []
    window.map_view.set_unit_selection([])


def _select_key(window, key) -> None:
    window._selection = [key]
    window._refresh_selection_view()


# (name, owner, const, tile). The three villagers sit on the tiles around corner (44, 44),
# where four chunks meet; the stroke at tile (44, 44) moves that corner.
_CHUNKED_FIXTURE = (
    ("town centre", 1, _TOWN_CENTRE, (50, 50)),
    ("house", 2, _HOUSE, (58, 50)),
    ("gate", 1, _GATE_NE, (60, 40)),
    ("villager 1", 1, _VILLAGER, (43, 43)),
    ("villager 2", 2, _VILLAGER, (44, 43)),
    ("tree", 0, _TREE, (43, 44)),
    # Indexed after villager 1 but earlier in its chunk's tile scan, so member order is tested.
    ("villager 3", 1, _VILLAGER, (42, 42)),
)


def _populate(window, view) -> dict:
    keys = {}
    for name, owner, const, tile in _CHUNKED_FIXTURE:
        keys[name] = _place_for(window, owner, const, tile)
        _assert_fresh(view, f"place {name}")
    _deselect(window)
    return keys


def _stroke(window, tile: tuple[int, int], size: int) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("elevation")
    window.brush_size_spin.setValue(size)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(*tile, Qt.NoModifier)
    window.on_edit_stroke_end()
    QApplication.processEvents()
    window.mode_combo.setCurrentText("Units")
    QApplication.processEvents()


def _move_across_a_chunk(window, keys) -> None:
    view = window.map_view
    entry = view._unit_index.entry_for_key(keys["town centre"])
    dx = _CHUNK - entry.own_x % _CHUNK
    window._move_units(window._ensure_unit_edits(), [entry], float(dx), 0.0, "Move unit")
    moved = view._unit_index.entry_for_key(keys["town centre"])
    assert moved.own_x // _CHUNK != entry.own_x // _CHUNK, "the move stayed in its chunk"


def _delete_house(window, keys) -> None:
    from PyQt5.QtCore import Qt

    _select_key(window, keys["house"])
    window.on_unit_delete(Qt.NoModifier)
    assert window.map_view._unit_index.entry_for_key(keys["house"]) is None, "the delete did not happen"


def _rotate_gate(window, keys) -> None:
    before = window.map_view._unit_index.entry_for_key(keys["gate"]).unit.unit_const
    _select_key(window, keys["gate"])
    window.on_unit_rotate(1)
    assert window.map_view._unit_index.entry_for_key(keys["gate"]).unit.unit_const != before, "no const change"


def _convert_villager(window, keys) -> None:
    entry = window.map_view._unit_index.entry_for_key(keys["villager 1"])
    window.units_panel.select_owner(2)
    window.brush_size_spin.setValue(1)
    window._begin_convert_stroke()
    window._convert_stroke_tile(entry.own_x, entry.own_y)
    window._end_convert_stroke()
    assert window.map_view._unit_index.entry_for_key(keys["villager 1"]) is None, "the convert did not happen"


def _raise_under_the_town_centre(window, keys) -> None:
    entry = window.map_view._unit_index.entry_for_key(keys["town centre"])
    _stroke(window, (entry.own_x, entry.own_y), 3)


def _raise_the_chunk_corner(window, keys) -> None:
    view = window.map_view
    entry = view._unit_index.entry_for_key(keys["villager 2"])
    before = view._unit_polygons_for(entry)
    _stroke(window, (44, 44), 1)
    if view._terrain_style == "sloped":
        assert view._unit_polygons_for(entry) != before, "the corner stroke moved no corner unit"


_CHUNKED_STEPS = (
    ("move across a chunk boundary", _move_across_a_chunk),
    ("delete", _delete_house),
    ("gate rotate", _rotate_gate),
    ("convert", _convert_villager),
    ("elevation stroke", _raise_under_the_town_centre),
    ("stroke at a chunk corner", _raise_the_chunk_corner),
)


@pytest.mark.parametrize("by_owner", [False, True], ids=["one-colour", "by-owner"])
@pytest.mark.parametrize("merged", [False, True], ids=["per-tile", "merged"])
@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_every_edit_and_its_undo_leave_each_group_equal_to_a_fresh_build(
    monkeypatch, style: str, merged: bool, by_owner: bool
) -> None:
    """The oracle: after every step each group item's subpaths, as a multiset,
    equal a from-scratch build of the same group."""
    from PyQt5.QtWidgets import QApplication

    window = _chunked_window(monkeypatch, style, merged, by_owner)
    try:
        view = window.map_view
        keys = _populate(window, view)
        if by_owner:
            assert len(view._footprint_items) == 3, "no owner groups to keep apart"
        for step, run in _CHUNKED_STEPS:
            run(window, keys)
            QApplication.processEvents()
            _assert_fresh(view, step)
            _assert_same_as_a_full_rebuild(view, step)
            _deselect(window)
            window.undo()
            QApplication.processEvents()
            _assert_fresh(view, f"undo {step}")
            _assert_same_as_a_full_rebuild(view, f"undo {step}")
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_one_unit_move_computes_only_its_old_and_new_chunks(monkeypatch) -> None:
    window = _chunked_window(monkeypatch)
    try:
        view = window.map_view
        keys = _populate(window, view)
        calls = _count_footprint_builds(monkeypatch, view)
        view.refresh_footprint_overlay()
        assert calls == [], "an unchanged refresh computed geometry"
        entry = view._unit_index.entry_for_key(keys["town centre"])
        old_chunk = (entry.own_x // _CHUNK, entry.own_y // _CHUNK)
        _move_across_a_chunk(window, keys)
        moved = view._unit_index.entry_for_key(keys["town centre"])
        chunks = {old_chunk, (moved.own_x // _CHUNK, moved.own_y // _CHUNK)}
        in_scope = unit_pick.footprint_entries(view._unit_index, view._footprint_scope)
        members = [e for e in in_scope if (e.own_x // _CHUNK, e.own_y // _CHUNK) in chunks]
        assert len(calls) == len(members) < len(in_scope)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_dataset_switch_rebuilds_every_chunk(monkeypatch) -> None:
    """A data mod changes tile_span and the drape test for the same const, so its generation is context."""
    from descape import game_dataset

    window = _chunked_window(monkeypatch)
    try:
        view = window.map_view
        _populate(window, view)
        calls = _count_footprint_builds(monkeypatch, view)
        monkeypatch.setattr(game_dataset, "_generation", game_dataset.generation() + 1)
        view.refresh_footprint_overlay()
        assert len(calls) == len(unit_pick.footprint_entries(view._unit_index, view._footprint_scope))
    finally:
        conftest.close_window(window)


def _chunks_holding(view) -> set:
    return {chunk for item in view._footprint_items.values() for chunk in item.chunks()}


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_unit_deleted_with_outlines_off_leaves_no_stale_chunk_once_they_are_enabled(monkeypatch) -> None:
    from PyQt5.QtCore import Qt

    window = _chunked_window(monkeypatch)
    try:
        view = window.map_view
        keys = _populate(window, view)
        house = view._unit_index.entry_for_key(keys["house"])
        chunk = (house.own_x // _CHUNK, house.own_y // _CHUNK)
        assert [e for e in view._unit_index.entries if (e.own_x // _CHUNK, e.own_y // _CHUNK) == chunk] == [house]
        assert chunk in _chunks_holding(view)
        window.footprint_action.setChecked(False)
        _select_key(window, keys["house"])
        window.on_unit_delete(Qt.NoModifier)
        window.footprint_action.setChecked(True)
        assert chunk not in _chunks_holding(view), "the deleted house's outline is still drawn"
        _assert_fresh(view, "enabled after a hidden delete")
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_scope_change_that_drops_a_chunks_last_member_clears_that_chunk(monkeypatch) -> None:
    window = _chunked_window(monkeypatch)
    try:
        view = window.map_view
        keys = _populate(window, view)
        villager = view._unit_index.entry_for_key(keys["villager 2"])
        chunk = (villager.own_x // _CHUNK, villager.own_y // _CHUNK)
        assert chunk in _chunks_holding(view)
        window.footprint_scope_actions[unit_pick.FOOTPRINT_SCOPE_MULTITILE].setChecked(True)
        assert view._footprint_scope == unit_pick.FOOTPRINT_SCOPE_MULTITILE
        assert chunk not in _chunks_holding(view), "the villager's outline outlived its scope"
        _assert_fresh(view, "scope change")
    finally:
        conftest.close_window(window)


def _render(scene, source, scale: float):
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter

    image = QImage(max(1, int(source.width() * scale)), max(1, int(source.height() * scale)), QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    scene.render(painter, QRectF(image.rect()), source)
    painter.end()
    return image


def _assert_same_pixels_as_one_path_item(scene, item, source, label: str) -> None:
    from PyQt5.QtWidgets import QGraphicsPathItem

    reference = QGraphicsPathItem(item.path())
    reference.setPen(item.pen())
    reference.setZValue(item.zValue())
    for scale in (1.0, 0.37):
        chunked = _render(scene, source, scale)
        item.setVisible(False)
        scene.addItem(reference)
        single = _render(scene, source, scale)
        scene.removeItem(reference)
        item.setVisible(True)
        blank = _render(scene, source, scale)
        blank.fill(0)
        assert single != blank, f"{label} at {scale}: the reference drew nothing, so this proves nothing"
        assert chunked == single, f"{label} at {scale}: chunked pixels differ from one path item's"


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_chunked_outlines_paint_the_pixels_one_path_item_would(monkeypatch, style: str) -> None:
    """Every other scene item hidden, so only the outline is compared. The
    first check runs right after a Place outside the old bounds, through
    scene().render(), which culls by boundingRect(): before any visibility
    toggle of the item, which would re-index it."""
    window = _chunked_window(monkeypatch, style)
    try:
        view = window.map_view
        _populate(window, view)
        scene = view.scene()
        item = view._footprint_items[None]
        assert item.chunk_count() > 3
        old_bounds = item.boundingRect()
        old_chunks = set(item.chunks())
        _place_for(window, 1, _TOWN_CENTRE, (100, 100))
        _deselect(window)
        (new_chunk,) = set(item.chunks()) - old_chunks
        new_rect = item.chunks()[new_chunk].controlPointRect().adjusted(-4, -4, 4, 4)
        assert not new_rect.intersects(old_bounds), "the Place stayed inside the old bounds"
        shown = [other for other in scene.items() if other is not item and other.isVisible()]
        for other in shown:
            other.setVisible(False)
        _assert_same_pixels_as_one_path_item(scene, item, new_rect, "after an outside Place")
        source = item.boundingRect().adjusted(-8, -8, 8, 8)
        _assert_same_pixels_as_one_path_item(scene, item, source, "the fixture")
        for other in shown:
            other.setVisible(True)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_chunk_outside_the_old_bounds_is_found_by_the_scene_index() -> None:
    """A deep BSP index, so the new chunk lands in another leaf than the old
    bounds: without prepareGeometryChange() the scene would never paint it."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QColor, QPainterPath, QPen
    from PyQt5.QtWidgets import QGraphicsScene

    from descape.viewer_canvas import ChunkedPathItem

    conftest.ensure_qapp()
    scene = QGraphicsScene(0, 0, 100000, 100000)
    scene.setBspTreeDepth(10)
    item = ChunkedPathItem(QPen(QColor(255, 0, 0), 0))
    scene.addItem(item)
    near, far = QPainterPath(), QPainterPath()
    near.addRect(10, 10, 10, 10)
    far.addRect(90000, 90000, 10, 10)
    item.set_chunk((0, 0), near)
    assert scene.items(QRectF(5, 5, 20, 20)) == [item]
    item.set_chunk((9, 9), far)
    assert scene.items(QRectF(89995, 89995, 20, 20)) == [item]
    item.set_chunk((0, 0), None)
    assert scene.items(QRectF(5, 5, 20, 20)) == []
    item.clear()
    assert item.boundingRect().isEmpty() and scene.items(QRectF(89995, 89995, 20, 20)) == []


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_paint_strokes_only_the_chunks_the_exposed_rect_meets() -> None:
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
    from PyQt5.QtWidgets import QStyleOptionGraphicsItem

    from descape.viewer_canvas import ChunkedPathItem

    conftest.ensure_qapp()

    class _Recording(QPainter):
        def __init__(self, device) -> None:
            super().__init__(device)
            self.drawn: list = []

        def drawPath(self, path) -> None:
            self.drawn.append(path.controlPointRect().left())
            super().drawPath(path)

    item = ChunkedPathItem(QPen(QColor(255, 0, 0), 0))
    left, right = QPainterPath(), QPainterPath()
    left.addRect(0, 0, 10, 10)
    right.addRect(100, 0, 10, 10)
    item.set_chunks({(0, 0): left, (1, 0): right})
    assert item.boundingRect() == QRectF(0, 0, 110, 10)
    image = QImage(200, 50, QImage.Format_ARGB32)
    for exposed, expected in (
        (QRectF(0, 0, 10, 10), [0.0]),
        (QRectF(110, 0, 20, 10), [100.0]),  # only touches the right chunk's edge
        (QRectF(111.5, 0, 20, 10), [100.0]),  # inside the 2 px stroke margin
        (QRectF(120, 0, 20, 10), []),
        (QRectF(0, 0, 200, 50), [0.0, 100.0]),
    ):
        painter = _Recording(image)
        option = QStyleOptionGraphicsItem()
        option.exposedRect = exposed
        item.paint(painter, option)
        painter.end()
        assert sorted(painter.drawn) == expected, exposed


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_fish_rings_stay_equal_to_a_fresh_build_through_chunked_edits(monkeypatch, style: str) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from descape import outline_chunks, unit_kind

    monkeypatch.setattr(outline_chunks, "CHUNK_TILES", _CHUNK)
    window = _window_with_fish(style)
    try:
        view = window.map_view

        def check(step: str) -> None:
            fish = unit_kind.fish_consts()
            entries = [e for e in view._unit_index.entries if e.unit.unit_const in fish]
            drawn = _subpaths(view._fish_item.path())
            assert drawn == _subpaths(view._fish_ring_path(entries)), step
            assert drawn, step

        check("built")
        units = window.scenario.unit_manager.units[0]
        (whale,) = [u for u in units if u.unit_const == _WHALE]
        whale.x += 3
        window._after_unit_mutation()
        check("move across a chunk")
        units.remove(next(u for u in units if u.unit_const == _DORADO))
        window._after_unit_mutation()
        check("remove")
        before = _subpaths(view._fish_item.path())
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(int(whale.x), int(whale.y), Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        assert _subpaths(view._fish_item.path()) != before, "the stroke moved no ring"
        check("elevation stroke")
    finally:
        conftest.close_window(window)
