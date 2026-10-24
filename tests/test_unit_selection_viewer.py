"""Units mode, its hover/selection cues and its inspector page, driven
through a real offscreen ViewerWindow -- phase 3's P3-d and P3-e.

Same technique and rationale as tests/test_toolbar_params.py: assert on
QAction/state handles and on MapView's own item references, never on widget
isVisible() for widgets in a never-shown window.

The blank template carries no units, so anything needing a real unit
population injects a synthetic one into the loaded scenario's unit_manager
before entering Units mode. That is enough here because every assertion in
this file is about WIRING -- which mode reaches what, what gets cleared
when, whether the index is rebuilt. The pick maths itself is proven against
the compositor in tests/test_unit_pick.py, not here.

Every ViewerWindow() constructed here calls edit_history.mark_saved() before
close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS
from descape.unit_filter import GAIA_PLAYER_ID, UnitFilter
from descape.units_panel import _ROTATION_TOOLTIP

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TREE_CONST = min(TREE_UNIT_IDS)
_WALL_CONST = 117  # stone wall: a variant index the game derives, never editable
_BUILDING_CONST = next(uid for uid, (sx, sy) in BUILDING_TILE_SPANS.items() if sx == 4 and sy == 4)


@dataclass
class SyntheticUnit:
    x: float
    y: float
    unit_const: int
    reference_id: int
    rotation: float = 0.0
    z: float = 0.0
    garrisoned_in_id: int = -1


def _window(with_units: bool = True):
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    if with_units:
        units = window.scenario.unit_manager.units
        units[0].append(SyntheticUnit(x=4.5, y=4.5, unit_const=_TREE_CONST, reference_id=101))
        units[1].append(SyntheticUnit(x=8.5, y=8.5, unit_const=_BUILDING_CONST, reference_id=102))
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _selection_items(map_view) -> list:
    """Every live selection scene item: each group's fill, under-stroke (if
    any) and outline."""
    return [item for group in map_view._unit_select_groups.values() for item in group if item is not None]


def _select_first_unit(window):
    """Selects via the same path a click takes, but resolving the entry from
    the index rather than synthesising a QPointF -- the pixel maths is
    test_unit_pick.py's job, not this file's."""
    index = window.map_view._unit_index
    entry = index.entries[-1]
    window._selection = [(entry.player_id, entry.unit.reference_id)]
    window.map_view.set_unit_selection([entry])
    window.units_panel.show_unit(entry)
    return entry


# --- mode reachability -------------------------------------------------


def test_units_mode_exists_alongside_the_other_modes() -> None:
    """An exact list, not a membership check: the combo's order is what
    _LEFT_PAGE_FOR_MODE's page indices are kept in step with, so a mode
    appearing anywhere other than appended is worth failing on."""
    window = _window()
    try:
        modes = [window.mode_combo.itemText(i) for i in range(window.mode_combo.count())]
        assert modes == [
            "View",
            "Terrain",
            "Units",
            "Triggers",
            "Map Options",
            "Players",
            "Diplomacy",
            "Messages",
        ]
    finally:
        _close(window)


def test_entering_units_mode_builds_the_index_and_tells_map_view() -> None:
    window = _window()
    try:
        assert window.map_view._unit_index is None
        window.mode_combo.setCurrentText("Units")
        assert window.mode == "units"
        assert window.map_view._mode == "units"
        assert window.map_view._unit_index is not None
        assert len(window.map_view._unit_index.entries) == 2
    finally:
        _close(window)


def test_units_mode_survives_switching_to_sloped() -> None:
    """The inverse of the pin Track C5's Step 4 removed. Units mode used to
    be forced back to View on entering Sloped, because there was no pick
    backend there and selection would have been a mode that silently did
    nothing on every click. C5 landed that backend, so the switch must now
    KEEP the mode.

    The SELECTION still goes, and that is set_source()'s own long-standing
    contract ("set_source() means a new scenario or a style switch, and
    neither guarantees the old key still addresses anything"), not something
    C5 changed -- it is what a Flat or Stepped switch has always done. What
    C5 changed is that Sloped now takes exactly that same path instead of
    additionally forcing the mode back to View. So this asserts the two
    behave IDENTICALLY rather than writing Sloped's behaviour down twice:
    a divergence in either direction is the thing worth catching.
    """
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        _select_first_unit(window)
        window.terrain_style_combo.setCurrentText("Flat")
        reference = (window.mode, not window.map_view._unit_select_groups)

        window.mode_combo.setCurrentText("Units")
        _select_first_unit(window)
        window.terrain_style_combo.setCurrentText("Sloped")
        assert window.mode == "units"
        assert window.mode_combo.currentText() == "Units"
        assert window.map_view._mode == "units"
        assert window.map_view._unit_index is not None
        assert (window.mode, not window.map_view._unit_select_groups) == reference, (
            "Sloped no longer handles a style switch the way Flat does"
        )
    finally:
        _close(window)


def test_selecting_units_while_already_in_sloped_is_allowed() -> None:
    """The reverse of the test above, and a genuinely different code route:
    on_mode_changed runs first and calls _update_tool_enabled, which is
    where the old forcing happened mid-call. Kept, inverted, because that
    route is what would break first if the gate ever came back."""
    window = _window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        window.mode_combo.setCurrentText("Units")
        assert window.mode == "units"
        assert window.mode_combo.currentText() == "Units"
        assert window.map_view._unit_index is not None
    finally:
        _close(window)


def test_units_mode_survives_a_style_switch_and_rebuilds_its_index() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            assert window.mode == "units", f"{style} should keep Units mode reachable"
            assert window.map_view._unit_index is not None, f"{style} lost its pick index"
    finally:
        _close(window)


def test_units_mode_is_nodrag_under_select_and_move_and_hand_drag_under_pan() -> None:
    """TASK-252: Pan pans in Units mode too; Select and Move need the press."""
    from PyQt5.QtWidgets import QGraphicsView

    window = _window()
    try:
        assert window.map_view.dragMode() == QGraphicsView.ScrollHandDrag
        window.mode_combo.setCurrentText("Units")
        assert window._current_tool == "pan"
        assert window.map_view.dragMode() == QGraphicsView.ScrollHandDrag
        for action in (window.select_action, window.move_action):
            action.setChecked(True)
            assert window.map_view.dragMode() == QGraphicsView.NoDrag, action.text()
        window.pan_action.setChecked(True)
        assert window.map_view.dragMode() == QGraphicsView.ScrollHandDrag
        window.select_action.setChecked(True)
        window.mode_combo.setCurrentText("View")
        assert window._current_tool == "pan"
        assert window.map_view.dragMode() == QGraphicsView.ScrollHandDrag
    finally:
        _close(window)


# --- left panel --------------------------------------------------------


def test_left_panel_swaps_to_the_inspector_and_back() -> None:
    window = _window()
    try:
        assert window.left_stack.currentIndex() == 0
        window.mode_combo.setCurrentText("Units")
        assert window.left_stack.currentIndex() == 2
        window.mode_combo.setCurrentText("Triggers")
        assert window.left_stack.currentIndex() == 1, "the triggers branch must survive the units one"
        window.mode_combo.setCurrentText("Terrain")
        assert window.left_stack.currentIndex() == 7  # GH #56's own page
        window.mode_combo.setCurrentText("View")
        assert window.left_stack.currentIndex() == 0
    finally:
        _close(window)


def test_inspector_populates_and_clears() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        assert window.units_panel.unit_inspector_empty.isVisibleTo(window.left_stack)

        entry = _select_first_unit(window)
        assert not window.units_panel.unit_inspector_empty.isVisibleTo(window.left_stack)
        assert window.units_panel.unit_field_labels["reference_id"].text() == str(entry.unit.reference_id)
        assert window.units_panel.unit_field_editors["player"].currentData() == 1
        assert window.units_panel.unit_field_labels["unit_const"].text() == str(_BUILDING_CONST)
        assert window.units_panel.unit_field_labels["name"].text()
        # GH #148: name and Unit ID live in the header, above the field grid.
        assert window.units_panel.unit_header.isVisibleTo(window.left_stack)
        assert window.units_panel.unit_header.isAncestorOf(window.units_panel.unit_field_labels["unit_const"])
        # _BUILDING_CONST has a full stats row (real ground truth, pinned
        # independently in tests/test_unit_stats_table.py).
        assert window.units_panel.unit_stats_grid.isVisibleTo(window.left_stack)
        _hp_caption, hp_value = window.units_panel.unit_stat_rows["hp"]
        assert hp_value.isVisibleTo(window.left_stack)
        assert hp_value.text()

        window.units_panel.show_unit(None)
        assert window.units_panel.unit_inspector_empty.isVisibleTo(window.left_stack)
        assert window.units_panel.unit_field_labels["reference_id"].text() == ""
        # Clearing blanks the stat labels too, not just the scenario fields.
        assert not window.units_panel.unit_header.isVisibleTo(window.left_stack)
        assert not window.units_panel.unit_stats_grid.isVisibleTo(window.left_stack)
        assert hp_value.text() == ""
    finally:
        _close(window)


@pytest.mark.parametrize("operation", ["open", "close"])
@pytest.mark.parametrize("count", [1, 2])
def test_opening_or_closing_a_map_clears_the_units_inspector(operation, count, tmp_path) -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        panel = window.units_panel
        _select(window, *window.map_view._unit_index.entries[:count])
        assert len(window._selection) == count
        assert panel.unit_inspector_grid.isVisibleTo(window.left_stack)
        assert panel.unit_field_labels["name"].text()

        if operation == "open":
            window.load_scenario(_UNITS_FIXTURE)
            assert window.scenario.path == _UNITS_FIXTURE
        else:
            window.close_scenario()
            assert window.scenario is None

        assert window.mode == "units"
        assert window._selection == []
        assert not window.map_view._unit_select_groups
        assert panel.unit_inspector_empty.isVisibleTo(window.left_stack)
        assert panel.unit_inspector_empty.text() == "No unit selected"
        assert not panel.unit_inspector_grid.isVisibleTo(window.left_stack)
        assert not panel.unit_header.isVisibleTo(window.left_stack)
        assert not panel.unit_stats_grid.isVisibleTo(window.left_stack)
        assert all(label.text() == "" for label in panel.unit_field_labels.values())
        assert panel.placing_card.isVisibleTo(window.left_stack)
        window.show()
        QApplication.processEvents()
        assert panel.grab().save(str(tmp_path / "empty_units_inspector.png"))
    finally:
        _close(window)


def test_a_tree_shows_only_hit_points_no_combat_rows() -> None:
    """Const 349 (Oak tree, this file's _TREE_CONST) has hp=20 and neither a
    type_50 nor a creatable block -- the four combat rows must disappear
    outright rather than show empty."""
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        tree_entry = next(e for e in window.map_view._unit_index.entries if e.unit.unit_const == _TREE_CONST)
        window.units_panel.show_unit(tree_entry)

        assert window.units_panel.unit_stats_grid.isVisibleTo(window.left_stack)
        _hp_caption, hp_value = window.units_panel.unit_stat_rows["hp"]
        assert hp_value.isVisibleTo(window.left_stack)
        assert hp_value.text() == "20"
        for field_id in ("attack", "melee_armour", "pierce_armour", "range"):
            caption, value = window.units_panel.unit_stat_rows[field_id]
            assert not caption.isVisibleTo(window.left_stack)
            assert not value.isVisibleTo(window.left_stack)
    finally:
        _close(window)


def test_loading_an_install_path_refreshes_the_unit_cards(monkeypatch) -> None:
    """TASK-337: the pending pick is sticky, so without this its card waits for the next pick."""
    from descape import asset_source

    dialog, window = conftest.dialog_and_window()
    try:
        calls: list[bool] = []
        monkeypatch.setattr(asset_source, "validate_install_path", lambda path: (True, "ok"))
        monkeypatch.setattr(asset_source, "set_install_path_override", lambda path: None)
        monkeypatch.setattr(asset_source, "save_install_path_to_config", lambda path: None)
        monkeypatch.setattr(window.units_panel, "refresh_thumbnails", lambda: calls.append(True))
        dialog.install_path_edit.setText("/nonexistent/aoe2de")
        dialog.load_install_path()
        assert calls == [True]
    finally:
        dialog.close()
        _close(window)


def test_stats_caveat_names_civ_bonuses_and_tech_upgrades() -> None:
    """GH #148: no visible note line, but the full caveat -- naming both civ
    bonuses and trigger effects -- is one hover away on the stats grid, same
    pattern as the Rotation caption's own tooltip."""
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        _select_first_unit(window)
        assert "civ bonuses" in window.units_panel.unit_stats_grid.toolTip()
        assert "trigger effects" in window.units_panel.unit_stats_grid.toolTip()
    finally:
        _close(window)


def test_gaia_owner_is_labelled_gaia_not_player_0() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        gaia_entry = next(e for e in window.map_view._unit_index.entries if e.player_id == 0)
        window.units_panel.show_unit(gaia_entry)
        assert window.units_panel.unit_field_editors["player"].currentData() == GAIA_PLAYER_ID
    finally:
        _close(window)


def test_rotation_is_shown_raw_with_its_variant_index_warning() -> None:
    """A top-level AGENTS.md hard rule: for ~65% of GAIA objects `rotation`
    is a doodad graphic-variant index, not an angle. The panel is the first
    place that becomes user-visible, so it must show the raw value and say
    so in the value's tooltip (GH #147: no longer an inline note)."""
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        gaia_entry = next(e for e in window.map_view._unit_index.entries if e.player_id == 0)
        # Pinned to a wall: the fixture's GAIA tree has an editable variant
        # since GH #123, and a wall's index stays raw and read-only.
        gaia_entry.unit.unit_const = _WALL_CONST
        gaia_entry.unit.rotation = 37
        window.units_panel.show_unit(gaia_entry)
        assert window.units_panel.unit_field_labels["rotation"].text() == "37"
        # One unit, so no group note leads the caption's tooltip (GH #198).
        assert window.units_panel.unit_rotation_label.toolTip() == _ROTATION_TOOLTIP
        assert "variant index" in window.units_panel.unit_field_labels["rotation"].toolTip()
        # The "stored in radians" half stays reachable regardless of
        # const, via the Rotation caption's own tooltip -- see
        # units_panel.py's _ROTATION_TOOLTIP.
        assert "radians" in window.units_panel.unit_rotation_label.toolTip()
    finally:
        _close(window)


# --- filter interaction ------------------------------------------------


def test_a_filter_that_hides_the_selection_clears_it() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        gaia_entry = next(e for e in window.map_view._unit_index.entries if e.player_id == 0)
        window._selection = [(gaia_entry.player_id, gaia_entry.unit.reference_id)]
        window.map_view.set_unit_selection([gaia_entry])
        window.units_panel.show_unit(gaia_entry)
        assert bool(window.map_view._unit_select_groups)

        window.show_gaia_action.setChecked(False)
        assert window._selection == []
        assert not window.map_view._unit_select_groups
        assert window.units_panel.unit_inspector_empty.isVisibleTo(window.left_stack)
    finally:
        _close(window)


def test_a_filter_that_spares_the_selection_keeps_it() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        entry = _select_first_unit(window)  # the player-1 building
        window.show_gaia_action.setChecked(False)
        assert window._selection == [(entry.player_id, entry.unit.reference_id)]
        assert bool(window.map_view._unit_select_groups)
    finally:
        _close(window)


def test_filter_change_rebuilds_the_index_so_hidden_units_are_unpickable() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        assert len(window.map_view._unit_index.entries) == 2
        window.show_trees_action.setChecked(False)
        assert len(window.map_view._unit_index.entries) == 1
        assert all(e.unit.unit_const != _TREE_CONST for e in window.map_view._unit_index.entries)
    finally:
        _close(window)


# --- cue hygiene -------------------------------------------------------


@pytest.mark.parametrize("style", ["Flat", "Stepped"])
def test_highlight_subpaths_are_closed_so_every_edge_strokes(style: str) -> None:
    """QPainterPath.addPolygon() leaves the subpath OPEN, so stroking a
    4-point diamond draws only 3 edges. The selection cue's fill hides that;
    the hover cue is outline-only and showed a visibly broken diamond --
    found by rendering it offscreen and looking at the image.

    Asserted as 5 elements per polygon (MoveTo + 3 LineTo + the close)
    rather than by eye, so a future refactor that drops closeSubpath() fails
    here instead of shipping a gap-toothed outline.
    """
    window = _window()
    try:
        window.terrain_style_combo.setCurrentText(style)
        window.mode_combo.setCurrentText("Units")
        for entry in window.map_view._unit_index.entries:
            path = window.map_view._unit_path(entry)
            polygons = len(
                [i for i in range(path.elementCount()) if path.elementAt(i).type == 0]
            )
            assert path.elementCount() == 5 * polygons, (
                f"{style}: unclosed subpath -- {path.elementCount()} elements for "
                f"{polygons} polygon(s), expected {5 * polygons}"
            )
    finally:
        _close(window)


def test_hover_and_selection_stack_in_a_fixed_order_either_way_round() -> None:
    """There is no other setZValue in this codebase -- everything else
    stacks by scene insertion order, so two lazily-created items would
    otherwise stack by whichever the user triggered first."""
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        index = window.map_view._unit_index
        window.map_view._update_unit_hover(index.entries[0])
        window.map_view.set_unit_selection([index.entries[1]])
        assert min(i.zValue() for i in _selection_items(window.map_view)) > window.map_view._unit_hover_item.zValue()
    finally:
        _close(window)


def test_hover_memoizes_on_the_unit_key() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        entry = window.map_view._unit_index.entries[0]
        window.map_view._update_unit_hover(entry)
        item = window.map_view._unit_hover_item
        window.map_view._update_unit_hover(entry)
        assert window.map_view._unit_hover_item is item, "re-hovering the same unit rebuilt the item"
    finally:
        _close(window)


def test_the_selection_survives_mode_switches_and_is_drawn_only_where_live() -> None:
    """TASK-359: a mode switch keeps the keys. Terrain keeps the highlight
    too, View keeps the key but hides it, and back in Units the highlight
    and the inspector's single-unit fields both return."""
    window = _window()
    try:
        mv = window.map_view
        window.mode_combo.setCurrentText("Units")
        entry = _select_first_unit(window)
        key = (entry.player_id, entry.unit.reference_id)

        window.mode_combo.setCurrentText("Terrain")
        assert window._selection == [key]
        assert mv._unit_select_keys == [key] and mv._unit_select_groups

        window.mode_combo.setCurrentText("View")
        assert window._selection == [key]
        assert mv._unit_select_keys == [] and not mv._unit_select_groups

        window.mode_combo.setCurrentText("Units")
        assert window._selection == [key]
        assert mv._unit_select_keys == [key] and mv._unit_select_groups
        panel = window.units_panel
        assert panel.unit_field_labels["reference_id"].text() == str(entry.unit.reference_id)
        assert panel.unit_field_labels["reference_id"].isVisibleTo(window.left_stack)
    finally:
        _close(window)


def test_leaving_units_mode_clears_the_hover_and_hides_the_selection_only_where_dormant() -> None:
    window = _window()
    try:
        mv = window.map_view
        window.mode_combo.setCurrentText("Units")
        _select_first_unit(window)
        for mode in ("Terrain", "View"):
            window.mode_combo.setCurrentText("Units")
            mv._update_unit_hover(mv._unit_index.entries[0])
            assert mv._unit_hover_item is not None
            window.mode_combo.setCurrentText(mode)
            assert mv._unit_hover_item is None, f"{mode}: the hover outlived Units mode"
            assert bool(mv._unit_select_groups) is (mode == "Terrain"), mode
    finally:
        _close(window)


def test_leave_event_clears_hover_but_keeps_selection() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        index = window.map_view._unit_index
        window.map_view._update_unit_hover(index.entries[0])
        window.map_view.set_unit_selection([index.entries[1]])
        window.map_view.leaveEvent(None)
        assert window.map_view._unit_hover_item is None
        assert bool(window.map_view._unit_select_groups), "selection must survive the cursor leaving"
    finally:
        _close(window)


# --- TASK-359: the selection kept across mode switches -------------------


def _select(window, *entries) -> None:
    """A viewer selection, through the reconciliation point every real path uses."""
    window._selection = [(e.player_id, e.unit.reference_id) for e in entries]
    window._refresh_selection_view()


def test_a_selected_unit_an_undo_in_terrain_mode_removes_is_dropped() -> None:
    from PyQt5.QtCore import QPointF, Qt

    window = _window(with_units=False)
    try:
        mv = window.map_view
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        window.mode_combo.setCurrentText("Units")
        window.units_panel.select_object(_HOUSE_CONST)
        window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        tp = mv._tile_pixels
        window.on_unit_place(QPointF(20 * tp + tp // 2, 20 * tp + tp // 2), Qt.NoModifier)
        assert len(window._selection) == 1, "Place Unit selects what it placed"
        window.mode_combo.setCurrentText("Terrain")
        assert mv._unit_select_groups
        assert window.scatter_action.isEnabled()

        window.undo()
        assert window._selection == []
        assert mv._unit_select_keys == [] and not mv._unit_select_groups
        assert not window.scatter_action.isEnabled()

        window.mode_combo.setCurrentText("Units")
        assert window.units_panel.unit_inspector_empty.isVisibleTo(window.left_stack)
        assert window.units_panel.unit_inspector_empty.text() == "No unit selected"
    finally:
        _close(window)


def test_an_elevation_edit_in_terrain_moves_the_kept_highlight() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        mv = window.map_view
        window.scenario.unit_manager.units[1].append(
            SyntheticUnit(x=20.0, y=20.0, unit_const=_BUILDING_CONST, reference_id=102)
        )
        window.mode_combo.setCurrentText("Units")
        _select(window, _entry_for(window, 1))
        window.mode_combo.setCurrentText("Terrain")
        (group,) = mv._unit_select_groups.values()
        before = group[2].path().boundingRect()

        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(20, 20, Qt.NoModifier)
        window.on_edit_stroke_end()
        # Deferred to the event loop with the footprint outlines (MapView.refresh_elevation_overlays).
        QApplication.processEvents()

        (group,) = mv._unit_select_groups.values()
        after = group[2].path().boundingRect()
        assert after != before, "the elevation edit did not move the highlight"
        assert after == mv._unit_path(mv._unit_index.entry_for_key((1, 102))).boundingRect()
    finally:
        conftest.close_window(window)


def test_a_filter_toggle_in_terrain_drops_the_selected_unit_it_hides() -> None:
    window = _window()
    try:
        mv = window.map_view
        window.mode_combo.setCurrentText("Units")
        building = _entry_for(window, 1)
        _select(window, _entry_for(window, GAIA_PLAYER_ID), building)
        window.mode_combo.setCurrentText("Terrain")
        assert len(window._selection) == 2

        window.show_gaia_action.setChecked(False)
        key = (1, building.unit.reference_id)
        assert window._selection == [key]
        assert mv._unit_select_keys == [key]
        assert list(mv._unit_select_groups) == [_tint(window, 1)]
        assert window.scatter_action.isEnabled(), "the surviving selection still supplies an area"
    finally:
        _close(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
@pytest.mark.parametrize("tool", ["elevation", "set_level"])
@pytest.mark.parametrize("flush_last_step", [False, True])
@pytest.mark.parametrize("end", ["release", "tool_switch"])
def test_a_kept_selection_refreshes_once_after_an_elevation_drag(monkeypatch, style, tool, flush_last_step, end) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    held = {"value": False}
    monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if held["value"] else Qt.NoButton))
    try:
        mv = window.map_view
        window.scenario.unit_manager.units[1].append(
            SyntheticUnit(x=20.0, y=20.0, unit_const=_BUILDING_CONST, reference_id=102)
        )
        window.terrain_style_combo.setCurrentText(style)
        assert mv._terrain_style == style.lower()
        window.mode_combo.setCurrentText("Units")
        _select(window, _entry_for(window, 1))
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected(tool)
        window.elevation_level_spin.setValue(4)
        QApplication.processEvents()
        before = next(iter(mv._unit_select_groups.values()))[2].path()
        calls = []
        real_refresh = mv.refresh_unit_highlight

        def refresh():
            calls.append(mv._stroke_active)
            real_refresh()

        monkeypatch.setattr(mv, "refresh_unit_highlight", refresh)
        held["value"] = True
        mv.mousePressEvent(conftest.mouse_event(
            QEvent.MouseButtonPress, conftest.polygon_viewport_pos(mv, 20, 20), Qt.LeftButton, Qt.LeftButton
        ))
        QApplication.processEvents()
        for step in range(1, 8):
            mv.mouseMoveEvent(conftest.mouse_event(
                QEvent.MouseMove, conftest.polygon_viewport_pos(mv, 20 + step, 20), Qt.NoButton, Qt.LeftButton
            ))
            if step < 7 or flush_last_step:
                QApplication.processEvents()
        assert mv._stroke_active and window.edit_history.in_stroke, "the drag ended early"
        held["value"] = False
        if end == "release":
            mv.mouseReleaseEvent(conftest.mouse_event(
                QEvent.MouseButtonRelease, conftest.polygon_viewport_pos(mv, 27, 20), Qt.LeftButton, Qt.NoButton
            ))
        else:
            window._on_tool_selected("pan")
        QApplication.processEvents()
        QApplication.processEvents()  # A leftover timer must not rebuild it a second time.
        assert calls == [False], "refresh once after release, never between drag steps"
        assert window._selection == [(1, 102)]
        after = next(iter(mv._unit_select_groups.values()))[2].path()
        assert after != before, "the stroke did not move the highlight"
        assert after == mv._unit_path(mv._unit_index.entry_for_key((1, 102)))
    finally:
        held["value"] = False
        if window.map_view._stroke_active:
            window.map_view._end_stroke()
        conftest.close_window(window)


@pytest.mark.parametrize("reconcile", ["filter", "render"])
def test_losing_the_last_kept_key_disables_scatter_in_terrain(reconcile) -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        _select(window, _entry_for(window, GAIA_PLAYER_ID))
        window.mode_combo.setCurrentText("Terrain")
        assert window.scatter_action.isEnabled() and window._region is None
        if reconcile == "filter":
            window.show_gaia_action.setChecked(False)
        else:
            window.scenario.unit_manager.units[GAIA_PLAYER_ID].clear()
            window._render_current(reset_view=False)
        assert window._selection == []
        assert not window.scatter_action.isEnabled()
    finally:
        _close(window)


_TRIGGERS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def test_a_dormant_selection_never_draws_over_pick_from_maps_highlight() -> None:
    """In Triggers the selection layer is Pick from map's. A filter toggle
    while it is armed, and a global undo of a unit edit (which disarms it,
    as every history move does), prune the kept keys without drawing them."""
    from PyQt5.QtCore import Qt

    gen = conftest.load_verify_module("gen_trigger_fixture")
    window = conftest.shown_window(1100, 760)
    try:
        mv = window.map_view
        window.load_scenario(_TRIGGERS_FIXTURE)
        window.terrain_style_combo.setCurrentText("Flat")
        window.mode_combo.setCurrentText("Units")
        archer = (1, gen.REF_ARCHER_A)
        _select(window, window.map_view._unit_index.entry_for_key(archer))
        assert window.on_unit_nudge(1, 0, Qt.NoModifier), "the nudge is the unit edit undone below"

        window.mode_combo.setCurrentText("Triggers")
        window.trigger_panel.select_trigger(gen.REFERENCE_TRIGGER)
        window.trigger_panel.select_entry("condition", gen.DESTROY_CONDITION)
        window.arm_unit_picker((gen.REFERENCE_TRIGGER, "condition", gen.DESTROY_CONDITION, "unit_object", False))
        house = (1, gen.REF_HOUSE)
        assert mv._unit_select_keys == [house] and mv._unit_select_groups

        window.show_gaia_action.setChecked(False)
        assert window._unit_picker is not None
        assert mv._unit_select_keys == [house] and mv._unit_select_groups
        assert window._selection == [archer]

        window.undo()
        assert window._unit_picker is None, "a history move disarms the picker"
        assert mv._unit_select_keys == [] and not mv._unit_select_groups
        assert window._selection == [archer]

        window.mode_combo.setCurrentText("Units")
        assert mv._unit_select_keys == [archer]
    finally:
        conftest.close_window(window)


def test_a_style_switch_keeps_the_selections_highlight_in_units_and_terrain() -> None:
    """set_source() drops MapView's keys; the re-render re-pushes the kept selection."""
    window = _window()
    try:
        mv = window.map_view
        window.mode_combo.setCurrentText("Units")
        entry = _select_first_unit(window)
        key = (entry.player_id, entry.unit.reference_id)
        for mode, style in (("Units", "Flat"), ("Terrain", "Stepped")):
            window.mode_combo.setCurrentText(mode)
            window.terrain_style_combo.setCurrentText(style)
            assert window._selection == [key], mode
            assert mv._unit_select_keys == [key] and mv._unit_select_groups, f"{mode}: the highlight was lost"
    finally:
        _close(window)


def test_opening_or_closing_a_map_drops_a_dormant_selection() -> None:
    """The same file reopened, so the old key resolves in the new document
    and only the per-document reset drops it."""
    window = _window(with_units=False)
    try:
        mv = window.map_view

        def select_then_park():
            window.mode_combo.setCurrentText("Units")
            entry = _select_first_unit(window)
            key = (entry.player_id, entry.unit.reference_id)
            window.mode_combo.setCurrentText("Terrain")
            window.mode_combo.setCurrentText("View")
            assert window._selection == [key]
            return key

        window.load_scenario(_UNITS_FIXTURE)
        key = select_then_park()
        window.edit_history.mark_saved()
        window.load_scenario(_UNITS_FIXTURE)
        assert any(u.reference_id == key[1] for u in window.scenario.unit_manager.units[key[0]]), "precondition"
        window.mode_combo.setCurrentText("Units")
        assert window._selection == []
        assert not mv._unit_select_groups

        select_then_park()
        window.edit_history.mark_saved()
        window.close_scenario()
        assert window._selection == []
    finally:
        _close(window)


def test_sloped_produces_a_unit_highlight_at_the_units_own_rise() -> None:
    """Inverted by Track C5's Step 4. The highlight must not merely exist:
    it has to sit where the marker was painted, which on sloped ground is
    the unit's own interpolated rise, so this compares the path's own
    bounding box against unit_polygons' answer rather than just asserting
    non-None."""
    window = _window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        from descape import unit_pick

        index = unit_pick.build_index(window.scenario, UnitFilter())
        window.map_view.set_unit_index(index)
        entry = index.entries[0]
        window.map_view.set_unit_selection([entry])
        (group,) = window.map_view._unit_select_groups.values()
        item = group[2]

        corner_rise = window.map_view._sloped_cache().corner_rise
        polygons = unit_pick.unit_polygons(
            entry,
            "sloped",
            window.map_view._tile_pixels,
            window.map_view._map_width,
            window.map_view._map_height,
            None,
            window.map_view._iso_proj,
            corner_rise=corner_rise,
        )
        xs = [x for poly in polygons for x, _y in poly]
        ys = [y for poly in polygons for _x, y in poly]
        rect = item.path().boundingRect()
        assert (round(rect.left()), round(rect.top())) == (min(xs), min(ys))
        assert (round(rect.right()), round(rect.bottom())) == (max(xs), max(ys))
    finally:
        _close(window)


def test_pick_unit_at_reuses_a_tile_it_is_handed_in_sloped() -> None:
    """mouseMoveEvent resolves the terrain tile once for the hover cue and
    hands it to pick_unit_at(), which used to look the same pixel up through
    the Sloped pick plane a second time.

    Both halves are pinned here, against a real cache rather than a
    synthetic corner_rise: the threaded tile must not change the answer, and
    it must be the value occlusion is actually decided by. The 4x4 building
    is deliberate. A multi-tile footprint keeps membership on
    diamond_membership(), so terrain_tile moves nothing but occlusion.
    """
    window = _window()
    try:
        from PyQt5.QtCore import QPointF

        from descape import unit_pick

        window.terrain_style_combo.setCurrentText("Sloped")
        # Sprites off: only there does any later tile occlude (TASK-319), which the near-corner check needs.
        window.show_sprites_action.setChecked(False)
        mv = window.map_view
        index = unit_pick.build_index(window.scenario, UnitFilter())
        mv.set_unit_index(index)
        entry = next(e for e in index.entries if e.unit.unit_const == _BUILDING_CONST)
        polygons = unit_pick.unit_polygons(
            entry,
            "sloped",
            mv._tile_pixels,
            mv._map_width,
            mv._map_height,
            None,
            mv._iso_proj,
            corner_rise=mv._sloped_cache().corner_rise,
        )
        poly = polygons[0]  # one footprint diamond; its centre is inside it
        pos = QPointF(
            sum(x for x, _y in poly) / len(poly),
            sum(y for _x, y in poly) / len(poly),
        )
        assert mv.pick_unit_at(pos) is entry, "the building is not pickable at its own diamond centre"

        tile = mv._pick_tile(pos)
        assert tile is not None
        assert mv.pick_unit_at(pos, tile) is entry, "the threaded tile changed the answer"

        # The map's near corner outranks every footprint tile's depth key, so
        # handing it in must occlude the building, and None ("no terrain
        # here") must leave it unoccluded. Both fail if the tile is ignored.
        assert mv.pick_unit_at(pos, (0, mv._map_height - 1)) is None
        assert mv.pick_unit_at(pos, None) is entry
    finally:
        _close(window)


# --- selection coloured by owner ---------------------------------------


def _owner_window():
    """_window()'s GAIA tree and P1 building plus a P2 unit, in Units mode."""
    window = _window()
    window.scenario.unit_manager.units[2].append(
        SyntheticUnit(x=12.5, y=12.5, unit_const=_BUILDING_CONST, reference_id=103)
    )
    window.mode_combo.setCurrentText("Units")
    return window


def _entry_for(window, player_id: int):
    return next(e for e in window.map_view._unit_index.entries if e.player_id == player_id)


def _rgb(color) -> tuple[int, int, int]:
    return (color.red(), color.green(), color.blue())


def _tint(window, player_id: int) -> tuple[int, int, int]:
    """The player's sprite tint, the lookup render.py makes (TASK-115)."""
    from descape.unit_sprites import TEAM_COLORS

    return tuple(TEAM_COLORS[window.scenario.team_indices[player_id] % len(TEAM_COLORS)])


def test_a_player_1_selection_is_one_group_in_player_1s_colour() -> None:
    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, 1)])
        p1 = _tint(window, 1)
        assert list(mv._unit_select_groups) == [p1]
        fill, under, outline = mv._unit_select_groups[p1]
        assert _rgb(outline.pen().color()) == p1
        assert outline.pen().widthF() == mv.UNIT_SELECT_PEN_WIDTH
        assert _rgb(fill.brush().color()) == p1
        assert fill.brush().color().alpha() == mv.UNIT_SELECT_FILL_ALPHA
        assert under is not None, "no dark under-stroke while colouring by owner"
        assert under.pen().widthF() == mv.UNIT_SELECT_UNDERSTROKE_WIDTH
        assert under.pen().color().getRgb() == mv.UNIT_SELECT_UNDERSTROKE_RGBA
    finally:
        _close(window)


def test_a_reassigned_color_id_is_what_the_highlight_follows() -> None:
    """The file's ColorId, not the player slot: a file can give P1 any colour."""
    from descape.scenario_io import refresh_player_colors
    from descape.unit_sprites import TEAM_COLORS

    window = _window()
    window.scenario.unit_manager.units[2].append(
        SyntheticUnit(x=12.5, y=12.5, unit_const=_BUILDING_CONST, reference_id=103)
    )
    try:
        refresh_player_colors(window.scenario, {1: 5})
        window.mode_combo.setCurrentText("Units")
        window.map_view.set_unit_selection([_entry_for(window, 1)])
        (key,) = window.map_view._unit_select_groups
        assert key == TEAM_COLORS[5 + 1]
        assert key != TEAM_COLORS[1]
    finally:
        _close(window)


# AoE2:DE's spritecolors.json TeamColors, typed out so a drift in either source shows.
_GAME_PURPLE = (138, 19, 247)
_GAME_GREY = (102, 102, 102)


def test_the_highlight_is_the_games_team_colour_not_the_hand_tuned_palette() -> None:
    """TASK-115 (user, 2026-10-19): purple and grey owners highlight in the
    game's own team colours, the sprite tint, by default and under a stored
    colour override that swaps the two."""
    from descape.scenario_io import refresh_player_colors
    from descape.terrain_palette import PLAYER_COLOR_BY_ID

    purple_id, grey_id = 5, 6  # ColorId PURPLE, GRAY
    assert PLAYER_COLOR_BY_ID[purple_id] != _GAME_PURPLE and PLAYER_COLOR_BY_ID[grey_id] != _GAME_GREY, (
        "precondition: the hand-tuned palette differs, or this pins nothing"
    )
    window = _window()
    units = window.scenario.unit_manager.units
    units[6].append(SyntheticUnit(x=12.5, y=12.5, unit_const=_BUILDING_CONST, reference_id=106))
    units[7].append(SyntheticUnit(x=16.5, y=16.5, unit_const=_BUILDING_CONST, reference_id=107))
    try:
        assert window.scenario.team_indices[6] == purple_id + 1, "precondition: P6 defaults to purple"
        assert window.scenario.team_indices[7] == grey_id + 1, "precondition: P7 defaults to grey"
        window.mode_combo.setCurrentText("Units")
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, 6), _entry_for(window, 7)])
        assert set(mv._unit_select_groups) == {_GAME_PURPLE, _GAME_GREY}
        assert _rgb(mv._unit_select_groups[_GAME_PURPLE][2].pen().color()) == _GAME_PURPLE
        assert _rgb(mv._unit_select_groups[_GAME_GREY][0].brush().color()) == _GAME_GREY

        # A stored override moves the tint index, and the highlight with it.
        refresh_player_colors(window.scenario, {6: grey_id, 7: purple_id})
        window.mode_combo.setCurrentText("View")
        window.mode_combo.setCurrentText("Units")
        mv.set_unit_selection([_entry_for(window, 6)])
        assert list(mv._unit_select_groups) == [_GAME_GREY] == [_tint(window, 6)]
        mv.set_unit_selection([_entry_for(window, 7)])
        assert list(mv._unit_select_groups) == [_GAME_PURPLE] == [_tint(window, 7)]
    finally:
        _close(window)


def test_a_purple_owners_outline_and_rings_take_the_highlights_team_colour() -> None:
    """TASK-449: footprint outlines and range rings by owner use the same
    team colour as the selection, so a purple building shows one purple."""
    from descape.terrain_palette import PLAYER_COLOR_BY_ID

    purple_id = 5  # ColorId PURPLE
    assert PLAYER_COLOR_BY_ID[purple_id] != _GAME_PURPLE, "precondition: the hand-tuned palette differs"
    window = _window(with_units=False)
    window.scenario.unit_manager.units[6].append(
        SyntheticUnit(x=20.0, y=20.0, unit_const=_CASTLE_CONST, reference_id=106)
    )
    try:
        assert window.scenario.team_indices[6] == purple_id + 1, "precondition: P6 defaults to purple"
        window.mode_combo.setCurrentText("Units")
        for action in (
            window.footprint_action,
            window.footprint_by_owner_action,
            window.range_rings_action,
            window.range_rings_buildings_action,
            window.range_rings_by_owner_action,
        ):
            action.setChecked(True)
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, 6)])
        assert list(mv._unit_select_groups) == [_GAME_PURPLE]
        assert set(mv._footprint_items) == {None, _GAME_PURPLE}
        assert list(mv._range_ring_items) == [_GAME_PURPLE]
        assert list(mv._ring_layer_items) == [_GAME_PURPLE]
        for items in (mv._footprint_items, mv._range_ring_items, mv._ring_layer_items):
            assert _rgb(items[_GAME_PURPLE].pen().color()) == _GAME_PURPLE
    finally:
        _close(window)


def test_gaia_keeps_the_configured_colour_with_an_under_stroke() -> None:
    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, GAIA_PLAYER_ID)])
        assert list(mv._unit_select_groups) == [None]
        fill, under, outline = mv._unit_select_groups[None]
        assert outline.pen().color() == mv._unit_select_pen.color()
        assert fill.brush().color() == mv._unit_select_fill_color
        assert under is not None, "GAIA drops the under-stroke, so a mixed selection has uneven edges"
    finally:
        _close(window)


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0)])
def test_every_fill_sits_below_every_under_stroke_below_every_outline(order) -> None:
    """Explicit Z, not insertion order: a later group's fill must not wash
    over an earlier group's edge, whichever group was created first."""
    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in order])
        groups = list(mv._unit_select_groups.values())
        assert len(groups) == 3
        fills = [g[0].zValue() for g in groups]
        unders = [g[1].zValue() for g in groups]
        outlines = [g[2].zValue() for g in groups]
        assert max(fills) < min(unders)
        assert max(unders) < min(outlines)
        assert max(outlines) < mv.REGION_SELECT_Z
    finally:
        _close(window)


def test_mixed_owners_make_three_groups_and_reselecting_one_drops_the_rest() -> None:
    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (0, 1, 2)])
        p1 = _tint(window, 1)
        p2 = _tint(window, 2)
        assert set(mv._unit_select_groups) == {None, p1, p2}
        dropped = [i for key in (None, p2) for i in mv._unit_select_groups[key] if i is not None]

        mv.set_unit_selection([_entry_for(window, 1)])
        assert list(mv._unit_select_groups) == [p1]
        assert all(item.scene() is None for item in dropped), "a dropped group left items in the scene"
    finally:
        _close(window)


def test_toggle_off_is_one_configured_group_with_no_under_stroke() -> None:
    """Off = today's drawing: one unioned fill+outline pair in the configured
    colour, whoever owns the units."""
    from descape import settings

    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (0, 1, 2)])
        window.selection_owner_colour_action.setChecked(False)
        assert settings.get_selection_by_owner() is False
        assert list(mv._unit_select_groups) == [None]
        fill, under, outline = mv._unit_select_groups[None]
        assert under is None
        assert outline.pen().color() == mv._unit_select_pen.color()
        assert outline.pen().widthF() == mv.UNIT_SELECT_PEN_WIDTH
        assert fill.brush().color() == mv._unit_select_fill_color
        assert fill.pen().style() == 0  # Qt.NoPen
        assert len([i for i in mv.scene().items() if i.zValue() in (mv.UNIT_SELECT_UNDER_Z,)]) == 0

        window.selection_owner_colour_action.setChecked(True)
        assert len(mv._unit_select_groups) == 3
        assert all(group[1] is not None for group in mv._unit_select_groups.values())
    finally:
        _close(window)


def test_a_colour_edit_recolours_a_live_selection() -> None:
    from descape import player_fields
    from descape.unit_sprites import TEAM_COLORS

    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, 1)])
        spec = next(s for s in player_fields.specs_for(window.scenario) if s.field_id == "color")
        window.set_player_field(spec, 1, 5)
        (key,) = mv._unit_select_groups
        assert key == TEAM_COLORS[5 + 1]
        assert _rgb(mv._unit_select_groups[key][2].pen().color()) == TEAM_COLORS[5 + 1]
    finally:
        _close(window)


def test_the_marquee_keeps_the_configured_colour() -> None:
    from PyQt5.QtCore import QPoint

    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, 1)])
        mv._update_marquee(QPoint(0, 0), QPoint(40, 40))
        assert mv._marquee_item.pen().color() == mv._unit_select_pen.color()
        assert mv._marquee_item.brush().color() == mv._unit_select_fill_color
    finally:
        _close(window)


@pytest.mark.parametrize("style", ["Flat", "Sloped"])
def test_the_dark_under_stroke_follows_every_owner_group_in_flat_and_sloped(style: str) -> None:
    """TASK-115 1, the Sloped half beside its Flat control.

    Passes when, in `style`, a GAIA + P1 selection makes two groups, each
    with a dark under-stroke at the shared width and RGBA, tracing exactly
    the outline's own non-empty path."""
    window = _owner_window()
    try:
        # Unchecked first, or Flat renders through the Flat+Isometric path.
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText(style)
        mv = window.map_view
        assert window._render_style == mv._terrain_style == style.lower()
        mv.set_unit_selection([_entry_for(window, p) for p in (GAIA_PLAYER_ID, 1)])
        assert set(mv._unit_select_groups) == {None, _tint(window, 1)}
        for _fill, under, outline in mv._unit_select_groups.values():
            assert under is not None, f"no dark under-stroke in {style}"
            assert under.scene() is mv.scene()
            assert under.pen().widthF() == mv.UNIT_SELECT_UNDERSTROKE_WIDTH
            assert under.pen().color().getRgb() == mv.UNIT_SELECT_UNDERSTROKE_RGBA
            assert not outline.path().isEmpty()
            assert under.path() == outline.path()
    finally:
        _close(window)


def test_a_players_panel_recolour_shows_on_the_next_units_selection() -> None:
    """TASK-115 3, the reachable version: the selection is dormant in Players
    mode (TASK-359), so the recolour happens with none drawn.

    Passes when P1's colour is changed through the Players panel's own colour
    combo, and back in Units mode the kept P1 selection, and a fresh one, is
    one group keyed, outlined and filled in the new colour, not the old one;
    and after an undo in Players mode, both are back in the old colour.

    The mode-entry re-push alone is not what this pins: the edit-time push in
    _apply_player_color_change() covers the same path, so it fails only with
    both removed."""
    from descape.unit_sprites import TEAM_COLORS

    window = _owner_window()
    try:
        mv = window.map_view
        p1 = _entry_for(window, 1)
        window._selection = [(1, p1.unit.reference_id)]
        window._refresh_selection_view()
        old = _tint(window, 1)
        assert list(mv._unit_select_groups) == [old]

        window.mode_combo.setCurrentText("Players")
        assert mv._unit_select_groups == {}
        assert window._selection == [(1, p1.unit.reference_id)], "kept while dormant"
        panel = window.players_panel
        assert panel.select_player(1)
        widget = panel.widget_for("color")
        # The combo's value is a ColorId; its tint is TEAM_COLORS one up (GAIA-first).
        target = next(
            i for i in range(widget.count())
            if widget.itemData(i) is not None and TEAM_COLORS[widget.itemData(i) + 1] != old
        )
        new = TEAM_COLORS[widget.itemData(target) + 1]
        widget.setCurrentIndex(target)
        assert window.edit_history.is_dirty

        window.mode_combo.setCurrentText("Units")
        assert list(mv._unit_select_groups) == [new], "the kept selection came back in the old colour"
        mv.set_unit_selection([_entry_for(window, 1)])
        assert list(mv._unit_select_groups) == [new]
        fill, _under, outline = mv._unit_select_groups[new]
        assert _rgb(outline.pen().color()) == new
        assert _rgb(fill.brush().color()) == new

        window.mode_combo.setCurrentText("Players")
        window.undo()
        assert not window.edit_history.is_dirty
        window.mode_combo.setCurrentText("Units")
        assert list(mv._unit_select_groups) == [old]
        mv.set_unit_selection([_entry_for(window, 1)])
        assert list(mv._unit_select_groups) == [old]
    finally:
        _close(window)


def test_a_settings_swatch_click_reinks_gaia_the_fill_and_the_marquee(monkeypatch) -> None:
    """TASK-115 4: the Settings > Appearance rows, not settings + apply_overlay_colors().

    Passes when, with a GAIA + P1 selection and a live marquee, clicking the
    Selection outline swatch (colour dialog stubbed to #ff00ff) re-inks
    GAIA's group outline and the marquee's pen, and clicking the Selection
    fill swatch (stubbed to #00ff80) re-inks GAIA's fill and the marquee's
    brush at the fixed fill alpha, while P1's owner group keeps P1's colour."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QColorDialog

    from descape import settings
    from descape.viewer import SettingsDialog

    window = _owner_window()
    dialog = SettingsDialog(window)
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (GAIA_PLAYER_ID, 1)])
        mv._update_marquee(QPoint(0, 0), QPoint(40, 40))
        p1 = _tint(window, 1)
        picks = iter([QColor("#ff00ff"), QColor("#00ff80")])
        monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: next(picks)))

        dialog._overlay_swatches["unit_select"].click()
        assert settings.get_overlay_color("unit_select") == "#ff00ff"
        assert _rgb(mv._unit_select_groups[None][2].pen().color()) == (255, 0, 255)
        assert _rgb(mv._marquee_item.pen().color()) == (255, 0, 255)

        dialog._overlay_swatches["unit_select_fill"].click()
        assert settings.get_overlay_color("unit_select_fill") == "#00ff80"
        gaia_fill = mv._unit_select_groups[None][0].brush().color()
        assert (_rgb(gaia_fill), gaia_fill.alpha()) == ((0, 255, 128), mv.UNIT_SELECT_FILL_ALPHA)
        marquee_fill = mv._marquee_item.brush().color()
        assert (_rgb(marquee_fill), marquee_fill.alpha()) == ((0, 255, 128), mv.UNIT_SELECT_FILL_ALPHA)
        assert _rgb(mv._marquee_item.pen().color()) == (255, 0, 255)

        p1_fill, _under, p1_outline = mv._unit_select_groups[p1]
        assert _rgb(p1_outline.pen().color()) == p1
        assert _rgb(p1_fill.brush().color()) == p1
    finally:
        dialog.close()
        _close(window)


def test_an_appearance_change_reinks_only_the_configured_group() -> None:
    from descape import settings

    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (0, 1)])
        p1 = _tint(window, 1)
        settings.set_overlay_color("unit_select", "#ff00ff")
        mv.apply_overlay_colors()
        assert _rgb(mv._unit_select_groups[None][2].pen().color()) == (255, 0, 255)
        assert _rgb(mv._unit_select_groups[p1][2].pen().color()) == p1
    finally:
        _close(window)


def test_close_then_reopen_leaves_no_stale_group_refs() -> None:
    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (0, 1)])
        window.edit_history.mark_saved()
        window.close_scenario()
        assert mv._unit_select_groups == {}
        assert mv._team_colors is None

        window.load_scenario(BLANK_TEMPLATE_PATH)
        window.scenario.unit_manager.units[1].append(
            SyntheticUnit(x=8.5, y=8.5, unit_const=_BUILDING_CONST, reference_id=102)
        )
        # The load built the index in Units mode, and re-entry reuses it, so announce the append.
        window._after_unit_mutation()
        window.mode_combo.setCurrentText("View")
        window.mode_combo.setCurrentText("Units")
        mv.set_unit_selection([_entry_for(window, 1)])
        assert list(mv._unit_select_groups) == [_tint(window, 1)]
    finally:
        _close(window)


# --- range rings (GH #49) ----------------------------------------------

_CASTLE_CONST = 82  # 4x4, range 8.0 -> a 10-tile ring
_HOUSE_CONST = 70  # 2x2, range 0.0
_ARCHER_CONST = 4  # range 4.0, but not a building
_OAK_CONST = 349  # a GAIA tree: neither a building nor ranged


def _ring_window(enabled: bool = True, second_castle: bool = False):
    """One Castle, one House and one archer on P1 plus a GAIA oak, in Units
    mode, with View > Range Rings on unless asked otherwise."""
    window = _window(with_units=False)
    units = window.scenario.unit_manager.units
    # Every even-span building sits at `tile + span/2`, so a 4x4 and a 2x2
    # both land on a whole coordinate. Off-parity: tests/test_range_ring_centre.py.
    units[1].append(SyntheticUnit(x=20.0, y=20.0, unit_const=_CASTLE_CONST, reference_id=201))
    units[1].append(SyntheticUnit(x=40.0, y=40.0, unit_const=_HOUSE_CONST, reference_id=202))
    units[1].append(SyntheticUnit(x=60.5, y=60.5, unit_const=_ARCHER_CONST, reference_id=203))
    units[0].append(SyntheticUnit(x=80.5, y=80.5, unit_const=_OAK_CONST, reference_id=204))
    if second_castle:
        units[1].append(SyntheticUnit(x=90.0, y=20.0, unit_const=_CASTLE_CONST, reference_id=205))
    window.mode_combo.setCurrentText("Units")
    window.range_rings_action.setChecked(enabled)
    return window


def _entry_for_const(window, unit_const: int):
    return next(e for e in window.map_view._unit_index.entries if e.unit.unit_const == unit_const)


def _ring_paths(mv) -> list:
    """Every live ring group's path, one per colour group."""
    return [item.path() for item in mv._range_ring_items.values()]


def test_selecting_a_castle_draws_a_ring_in_the_configured_colour() -> None:
    from descape import settings

    window = _ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        assert list(mv._range_ring_items) == [None], "Colour by Owner is off by default"
        item = mv._range_ring_items[None]
        assert not item.path().isEmpty()
        assert item.zValue() == mv.RANGE_RING_Z
        assert item.pen().color().name() == settings.get_overlay_color("range_ring")
        assert item.pen().widthF() == 0, "a non-cosmetic ring thins unevenly around the ellipse"
    finally:
        _close(window)


@pytest.mark.parametrize("unit_const", [_HOUSE_CONST, _ARCHER_CONST, _OAK_CONST])
def test_a_rangeless_building_and_a_non_building_draw_no_ring(unit_const: int) -> None:
    """The gate is both halves: a House is a building with range 0, an
    archer has range 4 but is not a building. A GAIA tree is GH #49's own case."""
    window = _ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, unit_const)])
        assert mv._unit_select_groups, "the selection itself should still be drawn"
        assert mv._range_ring_items == {}
    finally:
        _close(window)


def test_a_castle_and_an_archer_together_draw_exactly_one_ring() -> None:
    window = _ring_window()
    try:
        mv = window.map_view
        castle = _entry_for_const(window, _CASTLE_CONST)
        mv.set_unit_selection([castle])
        (path,) = _ring_paths(mv)
        alone = path.elementCount()
        mv.set_unit_selection([castle, _entry_for_const(window, _ARCHER_CONST)])
        (path,) = _ring_paths(mv)
        assert path.elementCount() == alone
    finally:
        _close(window)


def _subpath_count(path) -> int:
    return sum(1 for i in range(path.elementCount()) if path.elementAt(i).isMoveTo())


def test_two_selected_castles_draw_two_rings_and_deselecting_one_leaves_one() -> None:
    """GH #49's several-buildings step: one ring subpath per selected Castle."""
    window = _ring_window(second_castle=True)
    try:
        mv = window.map_view
        castles = [e for e in mv._unit_index.entries if e.unit.unit_const == _CASTLE_CONST]
        assert len(castles) == 2
        mv.set_unit_selection(castles)
        assert [_subpath_count(path) for path in _ring_paths(mv)] == [2]
        mv.set_unit_selection(castles[1:])
        assert [_subpath_count(path) for path in _ring_paths(mv)] == [1]
    finally:
        _close(window)


_UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def test_nudging_a_castle_moves_its_ring_one_tile_and_undo_puts_it_back() -> None:
    """GH #49's move + undo step. A real fixture, not SyntheticUnit: a nudge
    goes through UnitEditModel, whose construction gate needs real unit structs."""
    from PyQt5.QtCore import Qt

    from descape import unit_pick
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(_UNITS_FIXTURE)
    try:
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        window.range_rings_action.setChecked(True)
        model = window._ensure_unit_edits()
        assert model is not None
        with window._unit_edit(model, "Place castle", [1]):
            castle = model.add(1, _CASTLE_CONST, 30.0, 30.0)
        window.mode_combo.setCurrentText("Units")
        mv = window.map_view
        key = unit_pick.unit_key(1, castle)
        window._selection = [key]
        mv.set_unit_selection([mv._unit_index.entry_for_key(key)])
        (path,) = _ring_paths(mv)
        before = path.boundingRect().center()
        tp = mv._tile_pixels

        window.on_unit_nudge(1, 0, Qt.ShiftModifier)
        (path,) = _ring_paths(mv)
        moved = path.boundingRect().center()
        assert moved.x() - before.x() == pytest.approx(tp, abs=1.0)
        assert moved.y() == pytest.approx(before.y(), abs=1.0)

        window.undo()
        (path,) = _ring_paths(mv)
        restored = path.boundingRect().center()
        assert (restored.x(), restored.y()) == pytest.approx((before.x(), before.y()), abs=1.0)
    finally:
        _close(window)


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_every_terrain_style_draws_the_ring(style: str) -> None:
    """The three styles reach three different geometry branches (scene-space
    circle, Stepped's elevation rise, Sloped's corner_rise), so a missing
    height field shows up as a dropped ring rather than a misplaced one."""
    window = _ring_window()
    try:
        window.terrain_style_combo.setCurrentText(style)
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        paths = _ring_paths(mv)
        assert len(paths) == 1, f"{style}: no ring"
        assert not paths[0].isEmpty()
    finally:
        _close(window)


def test_the_ring_is_wider_than_the_buildings_own_footprint() -> None:
    """A 4x4 Castle with range 8 rings at 10 tiles, so the ring's bounding
    box has to dwarf the selection's -- the measured version of "a ring
    appeared"."""
    window = _ring_window()
    try:
        mv = window.map_view
        entry = _entry_for_const(window, _CASTLE_CONST)
        mv.set_unit_selection([entry])
        (path,) = _ring_paths(mv)
        ring = path.boundingRect()
        footprint = mv._unit_path(entry).boundingRect()
        assert ring.width() > 3 * footprint.width()
        assert ring.center().x() == pytest.approx(footprint.center().x(), abs=2.0)
        assert ring.center().y() == pytest.approx(footprint.center().y(), abs=2.0)
    finally:
        _close(window)


def test_deselecting_and_leaving_units_mode_both_clear_the_ring() -> None:
    window = _ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        mv.set_unit_selection(None)
        assert mv._range_ring_items == {}

        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        window.mode_combo.setCurrentText("View")
        assert mv._range_ring_items == {}
    finally:
        _close(window)


def test_the_toggle_drops_a_live_ring_and_puts_it_back() -> None:
    window = _ring_window(enabled=False)
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        assert mv._range_ring_items == {}, "no ring while View > Range Rings > Selected Buildings is off"

        window.range_rings_action.setChecked(True)
        (path,) = _ring_paths(mv)
        assert not path.isEmpty()

        window.range_rings_action.setChecked(False)
        assert mv._range_ring_items == {}
    finally:
        _close(window)


def test_the_toggle_persists_and_is_rebindable() -> None:
    from descape import settings

    window = _ring_window(enabled=False)
    try:
        assert settings.get_range_rings() is False
        window.range_rings_action.setChecked(True)
        assert settings.get_range_rings() is True
        assert ("view_range_rings", "Range Rings: Selected Buildings", "") in settings.REBINDABLE_ACTIONS
    finally:
        _close(window)


def test_a_colour_change_reinks_a_live_ring() -> None:
    from descape import settings

    window = _ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        settings.set_overlay_color("range_ring", "#ff00ff")
        mv.apply_overlay_colors()
        assert mv._range_ring_items[None].pen().color().name() == "#ff00ff"
    finally:
        _close(window)


def test_closing_the_scenario_leaves_no_stale_ring_ref() -> None:
    """scene().clear() destroys the C++ item; a surviving Python ref would
    raise RuntimeError on the next selection."""
    window = _ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        window.edit_history.mark_saved()
        window.close_scenario()
        assert mv._range_ring_items == {}
    finally:
        _close(window)


def test_a_style_switch_under_a_live_selection_rebuilds_the_ring() -> None:
    """set_source() nulls the ring item on a style switch; the selection is
    re-pushed afterwards, so the ring has to come back rather than leaving a
    dangling reference behind."""
    window = _ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        assert len(_ring_paths(mv)) == 1
        for style in ("Sloped", "Flat", "Stepped"):
            window.terrain_style_combo.setCurrentText(style)
            mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
            paths = _ring_paths(mv)
            assert len(paths) == 1, f"{style}: ring lost across the style switch"
            assert not paths[0].isEmpty()
    finally:
        _close(window)


# --- trigger range rings (GH #49 slice 2) -------------------------------


def _range_effect(value: float, unit_const: int = _CASTLE_CONST, *, player: int = 1, trigger_index: int = 0):
    """A Change Object Range SET `value` on `unit_const` for `player`."""
    from descape import trigger_range
    from descape.trigger_fields import UNSET

    return trigger_range.RangeEffect(
        sequence=trigger_index, trigger_index=trigger_index, trigger_name=f"T{trigger_index}", effect_index=0,
        effect_type=trigger_range.CHANGE_OBJECT_RANGE, effect_name="change_object_range",
        operation=trigger_range.SET, quantity=float(value), source_player=player, unit_const=unit_const,
        group=UNSET, object_type=UNSET, selected_ids=frozenset(), area=None,
        has_conditions=False, looping=False, enabled=True,
    )


def _trigger_ring_window(monkeypatch, effects, **kwargs):
    """_ring_window() with _trigger_range_effects() answering `effects` (a tuple or a sentinel)."""
    window = _ring_window(**kwargs)
    monkeypatch.setattr(window, "_trigger_range_effects", lambda: effects)
    return window


def _select_through_viewer(window, *entries) -> None:
    """The viewer's own selection path, which pushes the trigger effects first;
    mv.set_unit_selection() alone skips that push."""
    from descape import unit_pick

    window._selection = [unit_pick.unit_key(e.player_id, e.unit) for e in entries]
    window._refresh_selection_view()


def _trigger_paths(mv) -> list:
    return [item.path() for item in mv._trigger_ring_items.values()]


def _line(window) -> tuple[bool, str]:
    label = window.units_panel.trigger_range_label
    return (not label.isHidden(), label.text())


def test_a_trigger_set_draws_a_dashed_ring_beside_the_solid_one(monkeypatch) -> None:
    """SET 12 on the P1 Castle: radius 14 against the base ring's 10."""
    window = _trigger_ring_window(monkeypatch, (_range_effect(12),))
    try:
        mv = window.map_view
        _select_through_viewer(window, _entry_for_const(window, _CASTLE_CONST))
        assert list(mv._trigger_ring_items) == [None]
        dashed = mv._trigger_ring_items[None]
        (solid,) = _ring_paths(mv)
        ratio = dashed.path().boundingRect().width() / solid.boundingRect().width()
        assert ratio == pytest.approx(14 / 10, abs=0.01)
        assert dashed.path().boundingRect().center() == solid.boundingRect().center()
        assert dashed.pen().widthF() == 0
        assert dashed.pen().dashPattern() == [6.0, 4.0]
        assert dashed.zValue() == mv.RANGE_RING_Z
        assert _line(window) == (True, "Range with triggers: 12")
        assert "change object range, set 12" in window.units_panel.trigger_range_label.toolTip()
    finally:
        _close(window)


def test_a_house_with_base_range_zero_keeps_its_dashed_ring(monkeypatch) -> None:
    """The House has no solid ring; its dashed one must survive the empty base path set."""
    window = _trigger_ring_window(monkeypatch, (_range_effect(6, _HOUSE_CONST),))
    try:
        mv = window.map_view
        _select_through_viewer(window, _entry_for_const(window, _HOUSE_CONST))
        assert mv._range_ring_items == {}
        assert _subpath_count(mv._trigger_ring_items[None].path()) == 1
    finally:
        _close(window)


def test_trigger_changes_off_removes_only_the_dashed_ring(monkeypatch) -> None:
    window = _trigger_ring_window(monkeypatch, (_range_effect(12),))
    try:
        mv = window.map_view
        _select_through_viewer(window, _entry_for_const(window, _CASTLE_CONST))
        line = _line(window)
        window.range_rings_triggers_action.setChecked(False)
        assert mv._trigger_ring_items == {}
        assert len(_ring_paths(mv)) == 1
        assert _line(window) == line, "the Inspector line ignores the View toggles"
        window.range_rings_triggers_action.setChecked(True)
        assert len(_trigger_paths(mv)) == 1
    finally:
        _close(window)


def test_selected_buildings_off_removes_both_rings_and_keeps_the_line(monkeypatch) -> None:
    window = _trigger_ring_window(monkeypatch, (_range_effect(12),))
    try:
        mv = window.map_view
        _select_through_viewer(window, _entry_for_const(window, _CASTLE_CONST))
        line = _line(window)
        window.range_rings_action.setChecked(False)
        assert mv._range_ring_items == {} and mv._trigger_ring_items == {}
        assert _line(window) == line
        window.range_rings_action.setChecked(True)
        assert len(_ring_paths(mv)) == 1 and len(_trigger_paths(mv)) == 1
    finally:
        _close(window)


def test_unread_triggers_draw_no_dashed_ring_and_show_the_hint(monkeypatch) -> None:
    from descape import trigger_range

    window = _trigger_ring_window(monkeypatch, trigger_range.UNREAD)
    try:
        _select_through_viewer(window, _entry_for_const(window, _CASTLE_CONST))
        assert window.map_view._trigger_ring_items == {}
        assert _line(window) == (True, "Trigger range changes not read yet: open Triggers mode once")
    finally:
        _close(window)


def test_an_unreadable_file_shows_the_unreadable_line() -> None:
    import dataclasses

    window = _ring_window()
    try:
        window.scenario = dataclasses.replace(window.scenario, trigger_read_supported=False, _trigger_manager=None)
        _select_through_viewer(window, _entry_for_const(window, _CASTLE_CONST))
        assert _line(window) == (
            True, "This file's triggers can't be read, so trigger range changes are not shown"
        )
    finally:
        _close(window)


def test_a_non_building_shows_no_line_even_unread(monkeypatch) -> None:
    from descape import trigger_range

    window = _trigger_ring_window(monkeypatch, trigger_range.UNREAD)
    try:
        _select_through_viewer(window, _entry_for_const(window, _ARCHER_CONST))
        assert _line(window)[0] is False
    finally:
        _close(window)


def test_a_group_selection_hides_the_line(monkeypatch) -> None:
    window = _trigger_ring_window(monkeypatch, (_range_effect(12),), second_castle=True)
    try:
        mv = window.map_view
        castles = [e for e in mv._unit_index.entries if e.unit.unit_const == _CASTLE_CONST]
        _select_through_viewer(window, castles[0])
        assert _line(window)[0] is True
        _select_through_viewer(window, *castles)
        assert _line(window)[0] is False
        assert [_subpath_count(path) for path in _trigger_paths(mv)] == [2]
    finally:
        _close(window)


def test_colour_by_owner_keys_the_dashed_ring_on_the_owners_colour(monkeypatch) -> None:
    window = _trigger_ring_window(monkeypatch, (_range_effect(12),))
    try:
        mv = window.map_view
        window.range_rings_by_owner_action.setChecked(True)
        _select_through_viewer(window, _entry_for_const(window, _CASTLE_CONST))
        p1 = _tint(window, 1)
        assert list(mv._trigger_ring_items) == [p1]
        pen = mv._trigger_ring_items[p1].pen()
        assert _rgb(pen.color()) == p1
        assert pen.dashPattern() == [6.0, 4.0] and pen.widthF() == 0
    finally:
        _close(window)


def test_a_colour_change_reinks_the_dashed_ring_and_keeps_the_dash(monkeypatch) -> None:
    from descape import settings

    window = _trigger_ring_window(monkeypatch, (_range_effect(12),))
    try:
        mv = window.map_view
        _select_through_viewer(window, _entry_for_const(window, _CASTLE_CONST))
        settings.set_overlay_color("range_ring", "#ff00ff")
        mv.apply_overlay_colors()
        pen = mv._trigger_ring_items[None].pen()
        assert pen.color().name() == "#ff00ff"
        assert pen.dashPattern() == [6.0, 4.0]
    finally:
        _close(window)


def test_closing_the_scenario_drops_the_dashed_ring_refs_and_the_effects(monkeypatch) -> None:
    window = _trigger_ring_window(monkeypatch, (_range_effect(12),))
    try:
        mv = window.map_view
        _select_through_viewer(window, _entry_for_const(window, _CASTLE_CONST))
        assert mv._trigger_ring_items
        window.edit_history.mark_saved()
        window.close_scenario()
        assert mv._trigger_ring_items == {}
        assert mv._trigger_range_index is None
    finally:
        _close(window)


def test_trigger_changes_defaults_on_and_round_trips_through_the_file(monkeypatch) -> None:
    from descape import settings

    assert settings.get_range_rings_triggers() is True
    settings.set_range_rings_triggers(False)
    monkeypatch.setattr(settings, "_range_rings_triggers", None)
    assert settings.get_range_rings_triggers() is False
    settings.set_range_rings_triggers(True)
    monkeypatch.setattr(settings, "_range_rings_triggers", None)
    assert settings.get_range_rings_triggers() is True
    assert ("view_range_rings_triggers", "Range Rings: Trigger Changes", "") in settings.REBINDABLE_ACTIONS


def test_the_trigger_changes_toggle_persists() -> None:
    from descape import settings

    window = _ring_window()
    try:
        assert window.range_rings_triggers_action.isChecked() is True
        assert window.range_rings_triggers_action.text() == "&Trigger Changes"
        window.range_rings_triggers_action.setChecked(False)
        assert settings.get_range_rings_triggers() is False
        assert window.map_view._trigger_rings_enabled is False
    finally:
        _close(window)


_FIXTURE_HOUSE_REF = 502  # _TRIGGERS_FIXTURE's P1 House (const 70) at (41, 63)


def _trigger_fixture_window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(_TRIGGERS_FIXTURE)
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.range_rings_action.setChecked(True)
    return window


def _add_house_range_effect(window, value: int) -> None:
    """A modify_attribute MAXIMUM_RANGE SET for const 70 P1, in memory through the library."""
    from descape.scenario_io import parse_triggers

    manager = parse_triggers(window.scenario)
    manager.triggers[0].new_effect.modify_attribute(
        quantity=value, object_list_unit_id=_HOUSE_CONST, source_player=1, operation=1, object_attributes=12
    )


def _select_house(window) -> None:
    window.mode_combo.setCurrentText("Units")
    index = window.map_view._unit_index
    entry = index.entry_for_key((1, _FIXTURE_HOUSE_REF))
    assert entry is not None
    _select_through_viewer(window, entry)


def test_end_to_end_selection_never_forces_the_parse_then_a_real_effect_draws() -> None:
    window = _trigger_fixture_window()
    try:
        _select_house(window)
        assert window.scenario._trigger_manager is None, "selecting a building parsed the triggers"
        assert _line(window) == (True, "Trigger range changes not read yet: open Triggers mode once")

        window.mode_combo.setCurrentText("Triggers")
        _add_house_range_effect(window, 6)
        window._find_triggers_bumped()
        window._sync_trigger_ranges()
        _select_house(window)
        mv = window.map_view
        assert mv._range_ring_items == {}, "a House has base range 0: no solid ring"
        assert [_subpath_count(path) for path in _trigger_paths(mv)] == [1]
        assert _line(window) == (True, "Range with triggers: 6")
    finally:
        _close(window)


def test_a_trigger_field_commit_with_nothing_selected_walks_no_trigger(monkeypatch) -> None:
    from descape import trigger_range

    window = _trigger_fixture_window()
    try:
        window.mode_combo.setCurrentText("Triggers")
        calls = []
        real = trigger_range.collect_range_effects
        monkeypatch.setattr(trigger_range, "collect_range_effects", lambda *a: calls.append(1) or real(*a))
        assert not window._selection and not window.map_view._unit_select_keys
        window._after_trigger_edit("Edit field", None)
        assert calls == []

        _select_house(window)
        window.mode_combo.setCurrentText("Triggers")
        calls.clear()
        window._after_trigger_edit("Edit field", None)
        assert calls == [], "a kept selection is not drawn in Triggers mode: no walk"
        window.mode_combo.setCurrentText("Units")
        assert calls == [1], "the switch back to Units walks once"

        calls.clear()
        window._after_trigger_edit("Edit field", None)
        assert calls == [1], "with a drawn building selection the edit re-walks once"
    finally:
        _close(window)


def test_undo_and_redo_of_a_trigger_edit_follow_a_live_selection() -> None:
    window = _trigger_fixture_window()
    try:
        window.mode_combo.setCurrentText("Triggers")
        model = window._ensure_trigger_edits()
        with window._trigger_edit(model, "Add range effect", [0], refresh="entries"):
            model.manager().triggers[0].new_effect.modify_attribute(
                quantity=6, object_list_unit_id=_HOUSE_CONST, source_player=1, operation=1, object_attributes=12
            )
        _select_house(window)
        assert _line(window) == (True, "Range with triggers: 6")
        assert len(_trigger_paths(window.map_view)) == 1

        window.undo()
        assert window.map_view._trigger_ring_items == {}
        assert _line(window) == (False, "")
        window.redo()
        assert _line(window) == (True, "Range with triggers: 6")
        assert len(_trigger_paths(window.map_view)) == 1
    finally:
        _close(window)


@pytest.mark.parametrize("parser", ["find", "find_refs", "find_valid", "analysis"])
def test_a_parse_by_find_or_map_analysis_updates_a_live_selection(parser) -> None:
    """Each can parse while a Units selection is live; the hint must not outlive the parse."""
    window = _trigger_fixture_window()
    try:
        _select_house(window)
        assert _line(window)[1].startswith("Trigger range changes not read yet")
        if parser == "find":
            window._find_trigger_status()
        elif parser == "find_refs":
            window._find_trigger_refs_for([(1, _FIXTURE_HOUSE_REF)])
        elif parser == "find_valid":
            window._find_valid_trigger_indices({})
        else:
            window._show_analysis()
        assert window.scenario._trigger_manager is not None
        assert _line(window) == (False, "")
    finally:
        if window._analysis_dialog is not None:
            window._analysis_dialog.close()
        _close(window)


def test_loading_another_file_drops_the_cached_effects() -> None:
    """_trigger_gen is not per document: without the reset, the next file's
    first read at the same generation is served this file's effects."""
    from descape.scenario_io import parse_triggers

    window = _trigger_fixture_window()
    try:
        window.mode_combo.setCurrentText("Triggers")
        _add_house_range_effect(window, 6)
        effects = window._trigger_range_effects()
        assert len(effects) == 1
        gen = window._trigger_gen
        window.edit_history.mark_saved()
        window.load_scenario(_TRIGGERS_FIXTURE)
        window._trigger_gen = gen
        parse_triggers(window.scenario)
        assert window._trigger_range_effects() == ()
        assert window.map_view._trigger_range_effects is None
    finally:
        _close(window)


# --- range rings coloured by owner (GH #156) ---------------------------


def _owner_ring_window(by_owner: bool = True):
    """A P1, a P2 and a GAIA Castle in Units mode with View > Range Rings >
    Show on, and Colour by Owner on unless asked otherwise."""
    window = _window(with_units=False)
    units = window.scenario.unit_manager.units
    units[1].append(SyntheticUnit(x=20.0, y=20.0, unit_const=_CASTLE_CONST, reference_id=301))
    units[2].append(SyntheticUnit(x=60.0, y=20.0, unit_const=_CASTLE_CONST, reference_id=302))
    units[GAIA_PLAYER_ID].append(SyntheticUnit(x=20.0, y=60.0, unit_const=_CASTLE_CONST, reference_id=303))
    window.mode_combo.setCurrentText("Units")
    window.range_rings_action.setChecked(True)
    window.range_rings_by_owner_action.setChecked(by_owner)
    return window


def test_ring_owner_colour_defaults_off_and_the_ring_defaults_to_dark_gray(tmp_path) -> None:
    from descape import settings

    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_range_rings_by_owner() is False
    assert settings.get_overlay_color("range_ring") == "#404040"


def test_ring_owner_colour_round_trips_through_the_file(monkeypatch) -> None:
    from descape import settings

    settings.set_range_rings_by_owner(True)
    monkeypatch.setattr(settings, "_range_rings_by_owner", None)
    assert settings.get_range_rings_by_owner() is True
    settings.set_range_rings_by_owner(False)
    monkeypatch.setattr(settings, "_range_rings_by_owner", None)
    assert settings.get_range_rings_by_owner() is False


def test_range_rings_is_a_submenu_of_three_layers_and_colour_by_owner() -> None:
    from descape import settings

    window = _owner_ring_window(by_owner=False)
    try:
        view_menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
        range_menu = next(a.menu() for a in view_menu.actions() if a.text() == "&Range Rings")
        actions = range_menu.actions()
        assert actions[:5] == [
            window.range_rings_action,
            window.range_rings_triggers_action,
            window.range_rings_buildings_action,
            window.range_rings_selected_revealers_action,
            window.range_rings_revealers_action,
        ]
        assert actions[5].isSeparator()
        assert actions[6:] == [window.range_rings_by_owner_action]
        assert window.range_rings_revealers_action.text() == "All Map &Revealers"
        view_actions = view_menu.actions()
        position = {a: i for i, a in enumerate(view_actions)}
        ticks = next(a for a in view_actions if a.text() == "&Distance Ticks")
        assert position[window.trigger_overlay_action] < position[range_menu.menuAction()] < position[ticks]
        assert window.range_rings_action.text() == "&Selected Buildings"
        assert window.range_rings_action not in view_menu.actions(), "the toggle stayed in View itself"
        assert window.range_rings_by_owner_action.isChecked() is False

        window.range_rings_by_owner_action.setChecked(True)
        assert settings.get_range_rings_by_owner() is True
        assert window.map_view._range_rings_by_owner is True
        row = ("view_range_rings_by_owner", "Range Rings: Colour by Owner", "")
        assert row in settings.REBINDABLE_ACTIONS
    finally:
        _close(window)


def test_a_window_reads_ring_owner_colour_from_config_without_writing_it(tmp_path) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    config = tmp_path / "config.yaml"
    config.write_text("range_rings_by_owner: true\n")
    before = config.read_bytes()
    window = ViewerWindow()
    try:
        assert window.range_rings_by_owner_action.isChecked()
        assert window.map_view._range_rings_by_owner is True
        assert config.read_bytes() == before
    finally:
        _close(window)


def test_owner_colour_off_draws_every_owners_ring_in_the_one_configured_group() -> None:
    from descape import settings

    window = _owner_ring_window(by_owner=False)
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (GAIA_PLAYER_ID, 1, 2)])
        assert list(mv._range_ring_items) == [None]
        item = mv._range_ring_items[None]
        assert item.pen().color().name() == settings.get_overlay_color("range_ring")
        assert _subpath_count(item.path()) == 3
    finally:
        _close(window)


def test_a_player_1_castles_ring_is_player_1s_colour() -> None:
    window = _owner_ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, 1)])
        p1 = _tint(window, 1)
        assert list(mv._range_ring_items) == [p1]
        item = mv._range_ring_items[p1]
        assert _rgb(item.pen().color()) == p1
        assert _rgb(item.pen().color()) != _rgb(mv._range_ring_pen.color())
        assert item.pen().widthF() == 0, "a non-cosmetic ring thins unevenly around the ellipse"
        assert item.zValue() == mv.RANGE_RING_Z
        assert _subpath_count(item.path()) == 1
    finally:
        _close(window)


def test_a_gaia_castles_ring_keeps_the_configured_colour() -> None:
    window = _owner_ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, GAIA_PLAYER_ID)])
        assert list(mv._range_ring_items) == [None]
        assert mv._range_ring_items[None].pen().color() == mv._range_ring_pen.color()
    finally:
        _close(window)


def test_two_owners_draw_two_ring_groups_and_toggling_off_collapses_them() -> None:
    window = _owner_ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (1, 2)])
        p1 = _tint(window, 1)
        p2 = _tint(window, 2)
        assert p1 != p2
        assert set(mv._range_ring_items) == {p1, p2}
        assert [_subpath_count(path) for path in _ring_paths(mv)] == [1, 1]
        assert _rgb(mv._range_ring_items[p2].pen().color()) == p2
        owner_items = list(mv._range_ring_items.values())

        window.range_rings_by_owner_action.setChecked(False)
        assert list(mv._range_ring_items) == [None]
        assert [_subpath_count(path) for path in _ring_paths(mv)] == [2]
        assert all(item.scene() is None for item in owner_items), "a dropped group left its item in the scene"

        window.range_rings_by_owner_action.setChecked(True)
        assert set(mv._range_ring_items) == {p1, p2}
    finally:
        _close(window)


def test_a_colour_edit_recolours_a_live_ring() -> None:
    from descape import player_fields
    from descape.unit_sprites import TEAM_COLORS

    window = _owner_ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, 1)])
        (before,) = mv._range_ring_items
        assert before != TEAM_COLORS[5 + 1]
        spec = next(s for s in player_fields.specs_for(window.scenario) if s.field_id == "color")
        window.set_player_field(spec, 1, 5)
        (key,) = mv._range_ring_items
        assert key == TEAM_COLORS[5 + 1]
        assert _rgb(mv._range_ring_items[key].pen().color()) == TEAM_COLORS[5 + 1]
    finally:
        _close(window)


def test_an_appearance_change_reinks_only_the_configured_ring_group() -> None:
    from descape import settings

    window = _owner_ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (GAIA_PLAYER_ID, 1)])
        p1 = _tint(window, 1)
        assert set(mv._range_ring_items) == {None, p1}
        settings.set_overlay_color("range_ring", "#ff00ff")
        mv.apply_overlay_colors()
        assert _rgb(mv._range_ring_items[None].pen().color()) == (255, 0, 255)
        assert _rgb(mv._range_ring_items[p1].pen().color()) == p1
    finally:
        _close(window)


def test_closing_the_scenario_leaves_no_stale_owner_ring_refs() -> None:
    window = _owner_ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (1, 2)])
        assert len(mv._range_ring_items) == 2
        window.edit_history.mark_saved()
        window.close_scenario()
        assert mv._range_ring_items == {}
    finally:
        _close(window)


# --- always-on ring layer: All Buildings and Map Revealers (GH #153 / #155) ---


_MAP_REVEALER_CONST = 837  # LOS 4
_GIANT_MAP_REVEALER_CONST = 1775  # LOS 20
_TREBUCHET_CONST = 42  # mobile siege: a BUILDING_TILE_SPANS member, not a Show Buildings one


def _layer_window(*units):
    """Blank template in Flat, left in View mode (Units never entered), with
    `units` as (player_id, x, y, unit_const) and Footprint Outlines off."""
    from descape import settings

    window = _window(with_units=False)
    for ref, (player_id, x, y, unit_const) in enumerate(units, start=400):
        window.scenario.unit_manager.units[player_id].append(
            SyntheticUnit(x=x, y=y, unit_const=unit_const, reference_id=ref)
        )
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    assert window.mode == "view"
    assert settings.get_footprint_outlines() is False
    return window


def _layer_subpaths(mv, key=None) -> list:
    """One QPolygonF per ring in the layer's `key` group."""
    return list(mv._ring_layer_items[key].path().toSubpathPolygons())


def test_ring_layer_toggles_default_off_and_round_trip_through_the_file(monkeypatch) -> None:
    from descape import settings

    assert settings.get_range_rings_revealers() is False
    assert settings.get_range_rings_buildings() is False
    settings.set_range_rings_revealers(True)
    settings.set_range_rings_buildings(True)
    monkeypatch.setattr(settings, "_range_rings_revealers", None)
    monkeypatch.setattr(settings, "_range_rings_buildings", None)
    assert settings.get_range_rings_revealers() is True
    assert settings.get_range_rings_buildings() is True


def test_ring_layer_toggles_are_rebindable() -> None:
    from descape import settings

    assert ("view_range_rings_buildings", "Range Rings: All Buildings", "") in settings.REBINDABLE_ACTIONS
    assert ("view_range_rings_revealers", "Range Rings: All Map Revealers", "") in settings.REBINDABLE_ACTIONS
    row = ("view_range_rings_selected_revealers", "Range Rings: Selected Map Revealers", "")
    assert row in settings.REBINDABLE_ACTIONS


def test_map_revealers_ring_outside_units_mode_at_their_line_of_sight() -> None:
    from descape import settings

    window = _layer_window((0, 20.5, 20.5, _MAP_REVEALER_CONST), (0, 70.5, 70.5, _GIANT_MAP_REVEALER_CONST))
    try:
        mv = window.map_view
        assert mv._ring_layer_items == {}
        window.range_rings_revealers_action.setChecked(True)
        assert settings.get_range_rings_revealers() is True
        assert window.mode == "view", "the toggle must not need Units mode"
        assert list(mv._ring_layer_items) == [None]
        item = mv._ring_layer_items[None]
        assert item.zValue() == mv.RANGE_RING_Z
        assert item.pen().color().name() == settings.get_overlay_color("range_ring")
        tp = mv._tile_pixels
        widths = sorted(polygon.boundingRect().width() for polygon in _layer_subpaths(mv))
        assert widths == pytest.approx([2 * 4.0 * tp, 2 * 20.0 * tp], abs=1.0)

        window.range_rings_revealers_action.setChecked(False)
        assert mv._ring_layer_items == {}
        assert item.scene() is None
    finally:
        _close(window)


def test_all_buildings_rings_an_unselected_castle_but_not_a_trebuchet() -> None:
    window = _layer_window((1, 20.0, 20.0, _CASTLE_CONST), (1, 60.5, 60.5, _TREBUCHET_CONST))
    try:
        mv = window.map_view
        window.range_rings_buildings_action.setChecked(True)
        (ring,) = _layer_subpaths(mv)
        assert ring.boundingRect().width() == pytest.approx(2 * 10.0 * mv._tile_pixels, abs=1.0)
        assert mv._unit_select_keys == [], "nothing is selected"
    finally:
        _close(window)


def test_the_two_layer_toggles_are_independent() -> None:
    window = _layer_window((1, 20.0, 20.0, _CASTLE_CONST), (0, 70.5, 70.5, _MAP_REVEALER_CONST))
    try:
        mv = window.map_view
        window.range_rings_buildings_action.setChecked(True)
        assert len(_layer_subpaths(mv)) == 1
        window.range_rings_revealers_action.setChecked(True)
        assert len(_layer_subpaths(mv)) == 2
        window.range_rings_buildings_action.setChecked(False)
        (ring,) = _layer_subpaths(mv)
        assert ring.boundingRect().width() == pytest.approx(2 * 4.0 * mv._tile_pixels, abs=1.0)
    finally:
        _close(window)


def test_show_invisible_objects_off_hides_the_revealer_rings() -> None:
    window = _layer_window((0, 20.5, 20.5, _MAP_REVEALER_CONST))
    try:
        mv = window.map_view
        window.range_rings_revealers_action.setChecked(True)
        assert len(_layer_subpaths(mv)) == 1
        window.show_invisible_action.setChecked(False)
        assert mv._ring_layer_items == {}
        window.show_invisible_action.setChecked(True)
        assert len(_layer_subpaths(mv)) == 1
    finally:
        _close(window)


def test_colour_by_owner_splits_the_revealer_rings_by_player() -> None:
    window = _layer_window(
        (1, 20.5, 20.5, _MAP_REVEALER_CONST),
        (2, 60.5, 60.5, _MAP_REVEALER_CONST),
        (0, 90.5, 90.5, _MAP_REVEALER_CONST),
    )
    try:
        mv = window.map_view
        window.range_rings_revealers_action.setChecked(True)
        assert len(_layer_subpaths(mv)) == 3
        window.range_rings_by_owner_action.setChecked(True)
        p1 = _tint(window, 1)
        p2 = _tint(window, 2)
        assert set(mv._ring_layer_items) == {None, p1, p2}
        assert _rgb(mv._ring_layer_items[p1].pen().color()) == p1
        assert mv._ring_layer_items[p2].pen().widthF() == 0
        assert [len(_layer_subpaths(mv, key)) for key in (None, p1, p2)] == [1, 1, 1]
        window.range_rings_by_owner_action.setChecked(False)
        assert list(mv._ring_layer_items) == [None]
    finally:
        _close(window)


def test_an_appearance_change_reinks_the_layers_configured_group() -> None:
    from descape import settings

    window = _layer_window((0, 20.5, 20.5, _MAP_REVEALER_CONST))
    try:
        mv = window.map_view
        window.range_rings_revealers_action.setChecked(True)
        settings.set_overlay_color("range_ring", "#ff00ff")
        mv.apply_overlay_colors()
        assert mv._ring_layer_items[None].pen().color().name() == "#ff00ff"
    finally:
        _close(window)


def test_a_colour_edit_recolours_an_owner_coloured_layer_ring() -> None:
    from descape import player_fields
    from descape.unit_sprites import TEAM_COLORS

    window = _layer_window((1, 20.5, 20.5, _MAP_REVEALER_CONST))
    try:
        mv = window.map_view
        window.range_rings_revealers_action.setChecked(True)
        window.range_rings_by_owner_action.setChecked(True)
        (before,) = mv._ring_layer_items
        assert before != TEAM_COLORS[5 + 1]
        spec = next(s for s in player_fields.specs_for(window.scenario) if s.field_id == "color")
        window.set_player_field(spec, 1, 5)
        assert list(mv._ring_layer_items) == [TEAM_COLORS[5 + 1]]
    finally:
        _close(window)


def test_a_style_switch_and_closing_the_scenario_keep_the_layer_consistent() -> None:
    window = _layer_window((0, 20.5, 20.5, _MAP_REVEALER_CONST))
    try:
        mv = window.map_view
        window.range_rings_revealers_action.setChecked(True)
        window.mode_combo.setCurrentText("Terrain")
        for style in ("Sloped", "Stepped", "Flat"):
            old = mv._ring_layer_items[None]
            window.terrain_style_combo.setCurrentText(style)
            item = mv._ring_layer_items[None]
            assert item is not old, f"{style}: the layer was not rebuilt"
            assert item.scene() is mv.scene()
            assert not item.path().isEmpty()
        window.edit_history.mark_saved()
        window.close_scenario()
        assert mv._ring_layer_items == {}
    finally:
        _close(window)


def _fixture_window():
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(_UNITS_FIXTURE)
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    return window


def test_a_unit_edit_and_its_undo_in_terrain_mode_keep_the_revealer_rings() -> None:
    """Real UnitEditModel edits outside Units mode: each one rebuilds the
    index through _after_unit_mutation(), which has to land on the layer."""
    window = _fixture_window()
    try:
        window.mode_combo.setCurrentText("Terrain")
        window.range_rings_revealers_action.setChecked(True)
        mv = window.map_view
        before = len(_layer_subpaths(mv)) if mv._ring_layer_items else 0
        model = window._ensure_unit_edits()
        assert model is not None
        with window._unit_edit(model, "Place revealer", [0]):
            model.add(0, _MAP_REVEALER_CONST, 30.5, 30.5)
        with window._unit_edit(model, "Place revealer", [0]):
            model.add(0, _GIANT_MAP_REVEALER_CONST, 60.5, 60.5)
        assert window.mode == "terrain"
        assert len(_layer_subpaths(mv)) == before + 2
        window.undo()
        assert len(_layer_subpaths(mv)) == before + 1
    finally:
        _close(window)


def test_nudging_a_revealer_moves_its_layer_ring() -> None:
    """A Units-mode nudge patches the index in place (refresh_after_index_patch)."""
    from PyQt5.QtCore import Qt

    from descape import unit_pick

    window = _fixture_window()
    try:
        window.range_rings_revealers_action.setChecked(True)
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Place revealer", [0]):
            revealer = model.add(0, _MAP_REVEALER_CONST, 30.5, 30.5)
        window.mode_combo.setCurrentText("Units")
        mv = window.map_view
        key = unit_pick.unit_key(0, revealer)
        tp = mv._tile_pixels

        def ring_centre():
            # The ring around (30.5, 30.5): the subpath whose centre is nearest it.
            centres = [polygon.boundingRect().center() for polygon in _layer_subpaths(mv)]
            return min(centres, key=lambda c: abs(c.x() - 30.5 * tp) + abs(c.y() - 30.5 * tp))

        before = ring_centre()
        assert (before.x(), before.y()) == pytest.approx((30.5 * tp, 30.5 * tp), abs=1.0)
        window._selection = [key]
        mv.set_unit_selection([mv._unit_index.entry_for_key(key)])
        window.on_unit_nudge(1, 0, Qt.ShiftModifier)
        moved = ring_centre()
        assert (moved.x(), moved.y()) == pytest.approx((31.5 * tp, 30.5 * tp), abs=1.0)
    finally:
        _close(window)


def test_an_elevation_stroke_with_outlines_off_rebuilds_the_layer_once() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _layer_window((0, 20.5, 20.5, _MAP_REVEALER_CONST))
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        window.range_rings_revealers_action.setChecked(True)
        window.show()
        QApplication.processEvents()
        mv = window.map_view
        rebuilds = []
        real_refresh = mv.refresh_ring_layer
        mv.refresh_ring_layer = lambda: (rebuilds.append(1), real_refresh())[1]
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        for step in range(8):
            window.on_edit_stroke_tile(20 + step, 20, Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        assert len(rebuilds) == 1
        assert len(_layer_subpaths(mv)) == 1
    finally:
        mv.refresh_ring_layer = real_refresh
        _close(window)


def test_an_elevation_stroke_under_a_revealer_moves_its_ring_to_the_new_height() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _layer_window((0, 20.5, 20.5, _MAP_REVEALER_CONST))
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        window.range_rings_revealers_action.setChecked(True)
        window.show()
        QApplication.processEvents()
        mv = window.map_view
        (before,) = [polygon.boundingRect() for polygon in _layer_subpaths(mv)]
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(20, 20, Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        (after,) = [polygon.boundingRect() for polygon in _layer_subpaths(mv)]
        assert after.center().x() == pytest.approx(before.center().x())
        assert after.center().y() < before.center().y(), "the ring stayed at the old height"
    finally:
        _close(window)


def test_a_window_opened_with_a_ring_toggle_saved_on_draws_the_layer_without_entering_units_mode(tmp_path) -> None:
    from descape import settings
    from descape.edit_history import EditHistory
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel
    from descape.viewer import ViewerWindow

    loaded = load_map_and_units(_UNITS_FIXTURE)
    model = UnitEditModel(loaded)
    model.begin_unit_edit([1])
    model.add(1, _CASTLE_CONST, 60.0, 60.0)
    model.commit_unit_edit("Place castle", EditHistory())
    with_castle = tmp_path / "castle.aoe2scenario"
    write_scenario(loaded, with_castle, units=model, backup=False)

    settings.set_range_rings_buildings(True)
    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert window.range_rings_buildings_action.isChecked()
        window.load_scenario(with_castle)
        assert window.mode == "view"
        assert window.map_view._ring_layer_items.get(None) is not None
        assert _layer_subpaths(window.map_view)
    finally:
        _close(window)


def test_a_rebuild_reuses_unchanged_ring_polygons_and_a_style_switch_drops_them() -> None:
    window = _layer_window((0, 20.5, 20.5, _MAP_REVEALER_CONST), (1, 40.0, 40.0, _CASTLE_CONST))
    try:
        mv = window.map_view
        window.range_rings_revealers_action.setChecked(True)
        window.range_rings_buildings_action.setChecked(True)
        before = dict(mv._ring_layer_polygons)
        assert len(before) == 2
        mv.refresh_ring_layer()
        assert all(mv._ring_layer_polygons[key] is polygon for key, polygon in before.items())
        window.terrain_style_combo.setCurrentText("Stepped")
        assert len(mv._ring_layer_polygons) == 2
        assert not any(polygon is before.get(key) for key, polygon in mv._ring_layer_polygons.items())
        window.range_rings_revealers_action.setChecked(False)
        assert len(mv._ring_layer_polygons) == 1, "a ring no longer drawn is not kept"
    finally:
        _close(window)


# --- Selected Map Revealers ---------------------------------------------------


def _selected_revealer_window(selected_revealers: bool = True, selected_buildings: bool = False):
    """A P1 map revealer and Castle in Units mode, Flat, with only the selection toggles asked for."""
    window = _layer_window((1, 20.5, 20.5, _MAP_REVEALER_CONST), (1, 60.0, 60.0, _CASTLE_CONST))
    window.mode_combo.setCurrentText("Units")
    window.range_rings_selected_revealers_action.setChecked(selected_revealers)
    window.range_rings_action.setChecked(selected_buildings)
    return window


def test_selected_map_revealers_defaults_off_and_round_trips_through_the_file(monkeypatch) -> None:
    from descape import settings

    assert settings.get_range_rings_selected_revealers() is False
    settings.set_range_rings_selected_revealers(True)
    monkeypatch.setattr(settings, "_range_rings_selected_revealers", None)
    assert settings.get_range_rings_selected_revealers() is True


def test_a_selected_revealer_rings_at_its_line_of_sight() -> None:
    window = _selected_revealer_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _MAP_REVEALER_CONST)])
        (path,) = _ring_paths(mv)
        assert path.boundingRect().width() == pytest.approx(2 * 4.0 * mv._tile_pixels, abs=1.0)
        assert mv._ring_layer_items == {}, "the always-on layer stays off"
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        assert mv._range_ring_items == {}, "a Castle needs Selected Buildings"
    finally:
        _close(window)


def test_selected_buildings_alone_does_not_ring_a_selected_revealer() -> None:
    window = _selected_revealer_window(selected_revealers=False, selected_buildings=True)
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _MAP_REVEALER_CONST)])
        assert mv._range_ring_items == {}
    finally:
        _close(window)


def test_the_two_selection_toggles_ring_a_mixed_selection_independently() -> None:
    window = _selected_revealer_window(selected_buildings=True)
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, c) for c in (_MAP_REVEALER_CONST, _CASTLE_CONST)])
        assert [_subpath_count(path) for path in _ring_paths(mv)] == [2]
        window.range_rings_action.setChecked(False)
        (path,) = _ring_paths(mv)
        assert _subpath_count(path) == 1
        assert path.boundingRect().width() == pytest.approx(2 * 4.0 * mv._tile_pixels, abs=1.0)
        window.range_rings_selected_revealers_action.setChecked(False)
        assert mv._range_ring_items == {}
        window.range_rings_selected_revealers_action.setChecked(True)
        assert len(_ring_paths(mv)) == 1
    finally:
        _close(window)
