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

import pytest

import conftest
from descape.scenario_io import BLANK_TEMPLATE_PATH
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS
from descape.unit_filter import GAIA_PLAYER_ID, UnitFilter

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


def _select_first_unit(window):
    """Selects via the same path a click takes, but resolving the entry from
    the index rather than synthesising a QPointF -- the pixel maths is
    test_unit_pick.py's job, not this file's."""
    index = window.map_view._unit_index
    entry = index.entries[-1]
    window._selection = [(entry.player_id, entry.unit.reference_id)]
    window.map_view.set_unit_selection([entry])
    window._update_unit_inspector(entry)
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
        reference = (window.mode, window.map_view._unit_select_item is None)

        window.mode_combo.setCurrentText("Units")
        _select_first_unit(window)
        window.terrain_style_combo.setCurrentText("Sloped")
        assert window.mode == "units"
        assert window.mode_combo.currentText() == "Units"
        assert window.map_view._mode == "units"
        assert window.map_view._unit_index is not None
        assert (window.mode, window.map_view._unit_select_item is None) == reference, (
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
        assert window.left_stack.currentIndex() == 0
    finally:
        _close(window)


def test_inspector_populates_and_clears() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        assert window.unit_inspector_empty.isVisibleTo(window.left_stack)

        entry = _select_first_unit(window)
        assert not window.unit_inspector_empty.isVisibleTo(window.left_stack)
        assert window.unit_field_labels["reference_id"].text() == str(entry.unit.reference_id)
        assert window.unit_field_editors["player"].currentData() == 1
        assert window.unit_field_labels["unit_const"].text() == str(_BUILDING_CONST)
        assert window.unit_field_labels["name"].text()

        window._update_unit_inspector(None)
        assert window.unit_inspector_empty.isVisibleTo(window.left_stack)
        assert window.unit_field_labels["reference_id"].text() == ""
    finally:
        _close(window)


def test_gaia_owner_is_labelled_gaia_not_player_0() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        gaia_entry = next(e for e in window.map_view._unit_index.entries if e.player_id == 0)
        window._update_unit_inspector(gaia_entry)
        assert window.unit_field_editors["player"].currentData() == GAIA_PLAYER_ID
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
        window._update_unit_inspector(gaia_entry)
        assert window.unit_field_labels["rotation"].text() == "37"
        assert "variant index" in window.unit_rotation_note.text()
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
        window._update_unit_inspector(gaia_entry)
        assert window.map_view._unit_select_item is not None

        window.show_gaia_action.setChecked(False)
        assert window._selection == []
        assert window.map_view._unit_select_item is None
        assert window.unit_inspector_empty.isVisibleTo(window.left_stack)
    finally:
        _close(window)


def test_a_filter_that_spares_the_selection_keeps_it() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Units")
        entry = _select_first_unit(window)  # the player-1 building
        window.show_gaia_action.setChecked(False)
        assert window._selection == [(entry.player_id, entry.unit.reference_id)]
        assert window.map_view._unit_select_item is not None
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
        assert window.map_view._unit_select_item.zValue() > window.map_view._unit_hover_item.zValue()
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
        assert window.map_view._unit_select_item is None
        assert window.unit_inspector_empty.isVisibleTo(window.left_stack)
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
        assert window.map_view._unit_select_item is None
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
        assert window.map_view._unit_select_item is not None, "selection must survive the cursor leaving"
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
        item = window.map_view._unit_select_item
        assert item is not None

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
