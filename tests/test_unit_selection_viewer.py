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

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TREE_CONST = min(TREE_UNIT_IDS)
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


def test_units_mode_forces_nodrag_so_a_click_cannot_pan() -> None:
    from PyQt5.QtWidgets import QGraphicsView

    window = _window()
    try:
        assert window.map_view.dragMode() == QGraphicsView.ScrollHandDrag
        window.mode_combo.setCurrentText("Units")
        assert window.map_view.dragMode() == QGraphicsView.NoDrag
        # Pan is still nominally the active tool -- the mode, not the tool,
        # is what has to win here.
        assert window._current_tool == "pan"
        window.mode_combo.setCurrentText("View")
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
        # _BUILDING_CONST has a full stats row (real ground truth, pinned
        # independently in tests/test_unit_stats_table.py).
        assert window.units_panel.unit_stats_header.isVisibleTo(window.left_stack)
        _hp_caption, hp_value = window.units_panel.unit_stat_rows["hp"]
        assert hp_value.isVisibleTo(window.left_stack)
        assert hp_value.text()

        window.units_panel.show_unit(None)
        assert window.units_panel.unit_inspector_empty.isVisibleTo(window.left_stack)
        assert window.units_panel.unit_field_labels["reference_id"].text() == ""
        # Clearing blanks the stat labels too, not just the scenario fields.
        assert not window.units_panel.unit_stats_header.isVisibleTo(window.left_stack)
        assert hp_value.text() == ""
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

        assert window.units_panel.unit_stats_header.isVisibleTo(window.left_stack)
        _hp_caption, hp_value = window.units_panel.unit_stat_rows["hp"]
        assert hp_value.isVisibleTo(window.left_stack)
        assert hp_value.text() == "20"
        for field_id in ("attack", "melee_armour", "pierce_armour", "range"):
            caption, value = window.units_panel.unit_stat_rows[field_id]
            assert not caption.isVisibleTo(window.left_stack)
            assert not value.isVisibleTo(window.left_stack)
    finally:
        _close(window)


def test_stats_caveat_names_civ_bonuses_and_tech_upgrades() -> None:
    """The one-line note stays short at MIN_USEFUL_WIDTH, but the full
    caveat -- naming both civ bonuses and trigger effects -- is still one
    hover away, same pattern as the Rotation caption's own tooltip."""
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        _select_first_unit(window)
        assert "civ bonuses" in window.units_panel.unit_stats_note.toolTip()
        assert "trigger effects" in window.units_panel.unit_stats_note.toolTip()
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
    so."""
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        gaia_entry = next(e for e in window.map_view._unit_index.entries if e.player_id == 0)
        gaia_entry.unit.rotation = 37
        window.units_panel.show_unit(gaia_entry)
        assert window.units_panel.unit_field_labels["rotation"].text() == "37"
        assert "variant index" in window.units_panel.unit_rotation_note.text()
        assert window.units_panel.unit_rotation_note.isVisibleTo(window.units_panel)
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


def test_leaving_units_mode_clears_the_selection_key_not_just_the_cue() -> None:
    """MapView.set_mode() drops the visual selection on its own, so a stale
    ViewerWindow._selection would be invisible until something resolved it
    -- an inspector showing "No unit selected" while _selection still held a
    key. Phase 3.5 inherits _selection as the selection API; it must not
    lie."""
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        entry = _select_first_unit(window)
        assert window._selection == [(entry.player_id, entry.unit.reference_id)]

        window.mode_combo.setCurrentText("View")
        assert window._selection == []

        window.mode_combo.setCurrentText("Units")
        assert window._selection == []
        assert not window.map_view._unit_select_groups
        assert window.units_panel.unit_inspector_empty.isVisibleTo(window.left_stack)
    finally:
        _close(window)


def test_leaving_units_mode_clears_both_cues() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        index = window.map_view._unit_index
        window.map_view._update_unit_hover(index.entries[0])
        window.map_view.set_unit_selection([index.entries[1]])
        window.mode_combo.setCurrentText("View")
        assert window.map_view._unit_hover_item is None
        assert not window.map_view._unit_select_groups
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


def test_a_player_1_selection_is_one_group_in_player_1s_colour() -> None:
    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, 1)])
        p1 = tuple(window.scenario.player_colors[1])
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
    """player_colors, not PLAYER_COLORS: a file can give P1 any ColorId."""
    from descape.scenario_io import refresh_player_colors
    from descape.terrain_palette import PLAYER_COLOR_BY_ID, PLAYER_COLORS

    window = _window()
    window.scenario.unit_manager.units[2].append(
        SyntheticUnit(x=12.5, y=12.5, unit_const=_BUILDING_CONST, reference_id=103)
    )
    try:
        refresh_player_colors(window.scenario, {1: 5})
        window.mode_combo.setCurrentText("Units")
        window.map_view.set_unit_selection([_entry_for(window, 1)])
        (key,) = window.map_view._unit_select_groups
        assert key == PLAYER_COLOR_BY_ID[5]
        assert key != tuple(PLAYER_COLORS[1])
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
        p1 = tuple(window.scenario.player_colors[1])
        p2 = tuple(window.scenario.player_colors[2])
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
    from descape.terrain_palette import PLAYER_COLOR_BY_ID

    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, 1)])
        spec = next(s for s in player_fields.specs_for(window.scenario) if s.field_id == "color")
        window.set_player_field(spec, 1, 5)
        (key,) = mv._unit_select_groups
        assert key == PLAYER_COLOR_BY_ID[5]
        assert _rgb(mv._unit_select_groups[key][2].pen().color()) == PLAYER_COLOR_BY_ID[5]
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


def test_an_appearance_change_reinks_only_the_configured_group() -> None:
    from descape import settings

    window = _owner_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for(window, p) for p in (0, 1)])
        p1 = tuple(window.scenario.player_colors[1])
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
        assert mv._selection_player_colors is None

        window.load_scenario(BLANK_TEMPLATE_PATH)
        window.scenario.unit_manager.units[1].append(
            SyntheticUnit(x=8.5, y=8.5, unit_const=_BUILDING_CONST, reference_id=102)
        )
        window.mode_combo.setCurrentText("View")
        window.mode_combo.setCurrentText("Units")
        mv.set_unit_selection([_entry_for(window, 1)])
        assert list(mv._unit_select_groups) == [tuple(window.scenario.player_colors[1])]
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
    # both land on a whole coordinate -- the ring's centre and the footprint's
    # only coincide when the placement is the one real files use.
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


def test_selecting_a_castle_draws_a_ring_in_the_configured_colour() -> None:
    from descape import settings

    window = _ring_window()
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        item = mv._range_ring_item
        assert item is not None
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
        assert mv._range_ring_item is None
    finally:
        _close(window)


def test_a_castle_and_an_archer_together_draw_exactly_one_ring() -> None:
    window = _ring_window()
    try:
        mv = window.map_view
        castle = _entry_for_const(window, _CASTLE_CONST)
        mv.set_unit_selection([castle])
        alone = mv._range_ring_item.path().elementCount()
        mv.set_unit_selection([castle, _entry_for_const(window, _ARCHER_CONST)])
        assert mv._range_ring_item.path().elementCount() == alone
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
        assert _subpath_count(mv._range_ring_item.path()) == 2
        mv.set_unit_selection(castles[1:])
        assert _subpath_count(mv._range_ring_item.path()) == 1
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
        before = mv._range_ring_item.path().boundingRect().center()
        tp = mv._tile_pixels

        window.on_unit_nudge(1, 0, Qt.ShiftModifier)
        moved = mv._range_ring_item.path().boundingRect().center()
        assert moved.x() - before.x() == pytest.approx(tp, abs=1.0)
        assert moved.y() == pytest.approx(before.y(), abs=1.0)

        window.undo()
        restored = mv._range_ring_item.path().boundingRect().center()
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
        assert mv._range_ring_item is not None, f"{style}: no ring"
        assert not mv._range_ring_item.path().isEmpty()
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
        ring = mv._range_ring_item.path().boundingRect()
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
        assert mv._range_ring_item is None

        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        window.mode_combo.setCurrentText("View")
        assert mv._range_ring_item is None
    finally:
        _close(window)


def test_the_toggle_drops_a_live_ring_and_puts_it_back() -> None:
    window = _ring_window(enabled=False)
    try:
        mv = window.map_view
        mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
        assert mv._range_ring_item is None, "no ring while View > Range Rings is off"

        window.range_rings_action.setChecked(True)
        assert mv._range_ring_item is not None
        assert not mv._range_ring_item.path().isEmpty()

        window.range_rings_action.setChecked(False)
        assert mv._range_ring_item is None
    finally:
        _close(window)


def test_the_toggle_persists_and_is_rebindable() -> None:
    from descape import settings

    window = _ring_window(enabled=False)
    try:
        assert settings.get_range_rings() is False
        window.range_rings_action.setChecked(True)
        assert settings.get_range_rings() is True
        assert ("view_range_rings", "Range Rings", "") in settings.REBINDABLE_ACTIONS
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
        assert mv._range_ring_item.pen().color().name() == "#ff00ff"
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
        assert mv._range_ring_item is None
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
        assert mv._range_ring_item is not None
        for style in ("Sloped", "Flat", "Stepped"):
            window.terrain_style_combo.setCurrentText(style)
            mv.set_unit_selection([_entry_for_const(window, _CASTLE_CONST)])
            assert mv._range_ring_item is not None, f"{style}: ring lost across the style switch"
            assert not mv._range_ring_item.path().isEmpty()
    finally:
        _close(window)
