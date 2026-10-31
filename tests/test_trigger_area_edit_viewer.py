"""GH #229: Edit Area, moving and resizing the selected trigger area with the
mouse, through a real offscreen ViewerWindow on
tests/fixtures/triggers_120x120.aoe2scenario: the toolbar checkbox, the
gates that make the handles live, hover cursors, the drag and its one
undo record, and every cancel."""

from __future__ import annotations

import pytest
from test_trigger_tile_picker_viewer import (
    AREA,
    CREATE_EFFECT,
    HALF_SET_EFFECT,
    LOCATION,
    PATROL_EFFECT,
    _arm,
    _escape,
    _gen,
    _ground_pos,
    _move,
    _press,
    _records,
    _release,
    _values,
    _window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

STYLES = ["Flat", "Flat top-down", "Stepped", "Sloped"]


def _editing(style: str = "Flat"):
    """The patrol effect's area (30, 40)-(34, 46) shown alone, Edit Area ticked, Pan active."""
    window = _window(style)
    window.trigger_panel.select_entry("effect", PATROL_EFFECT)
    window.area_edit_check.setChecked(True)
    assert window.map_view.area_edit() == _gen().GEOMETRY_PATROL_AREA
    # Zoomed in: at the fit zoom one viewport pixel is a sixth of a tile.
    window.map_view.scale(4.0, 4.0)
    window.map_view.center_on_tile(32, 43)
    return window


def _point(view, zone: str):
    """A viewport point inside the patrol area, in `zone`'s grab band."""
    x1, y1, x2, y2 = _gen().GEOMETRY_PATROL_AREA
    band = min(view._area_band_tiles(), 5 / 3) / 2
    sides = "" if zone == "move" else zone
    mx = x1 + band if "w" in sides else x2 + 1 - band if "e" in sides else (x1 + x2 + 1) / 2
    my = y1 + band if "n" in sides else y2 + 1 - band if "s" in sides else (y1 + y2 + 1) / 2
    return _ground_pos(view, mx, my)


def _drag(view, start, end):
    _press(view, start)
    _move(view, end)
    _release(view, end)


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _mark_pos(view, tile=None):
    """The viewport point at a location mark's centre, the patrol's by default."""
    x, y = tile or _gen().GEOMETRY_PATROL_LOCATION
    return _ground_pos(view, x + 0.5, y + 0.5)


def _set_location(window, entry, tile) -> None:
    gen = _gen()
    values = dict(zip(LOCATION, tile, strict=True))
    window.set_entry_field_group(gen.REFERENCE_TRIGGER, ("effect", entry), values, "Set Location")


# -- the four checkboxes below Mode --------------------------------------------------


def _options_shown(window) -> list[bool]:
    return [action.isVisible() for action, check in window._area_option_params if check is not None]


def _options_enabled(window) -> list[bool]:
    return [check.isEnabled() for _action, check in window._area_option_params if check is not None]


def test_the_checkboxes_show_under_pan_in_triggers_mode_only() -> None:
    from descape import settings

    window = _window()
    try:
        assert window._current_tool == "pan"
        checks = [c for _a, c in window._area_option_params if c is not None]
        assert [c.text() for c in checks] == ["Edit Area", "Move", "Resize", "Tile borders"]
        assert [c.isChecked() for c in checks] == [False, True, True, False]
        assert _options_shown(window) == [True] * 4
        assert _options_enabled(window) == [True, False, False, False], "greyed while Edit Area is off"
        window.area_edit_check.setChecked(True)
        assert settings.get_tool_option_default("area_edit") is False, "a toolbar tick lasts the session only"
        assert _options_enabled(window) == [True] * 4
        window.ruler_action.setChecked(True)
        assert _options_shown(window) == [False] * 4, "they belong to Pan"
        assert window.map_view.area_edit() is None
        window.pan_action.setChecked(True)
        assert _options_shown(window) == [True] * 4
        window.mode_combo.setCurrentText("Units")
        assert _options_shown(window) == [False] * 4
        assert window.map_view.area_edit() is None
    finally:
        conftest.close_window(window)


def test_the_checkboxes_are_not_on_the_main_toolbar() -> None:
    window = _window()
    try:
        assert all(action not in window.main_toolbar.actions() for action, _c in window._area_option_params)
        assert [a.text() for a in window.main_toolbar.actions() if a.text() == "Edit Area"] == []
    finally:
        conftest.close_window(window)


def test_the_four_boxes_start_at_their_settings_tools_defaults() -> None:
    """TASK-272: Settings > Tools' defaults reach the boxes and the map view's
    options at construction; a toolbar tick changes no default."""
    from descape import settings

    settings.set_tool_option_defaults(
        {"area_edit": True, "area_move": False, "area_resize": True, "area_tile_borders": True}
    )
    window = _window()
    try:
        assert window.area_edit_check.isChecked()
        assert not window.area_move_check.isChecked()
        assert window.area_tile_borders_check.isChecked()
        assert (window.map_view._area_move, window.map_view._area_tile_borders) == (False, True)
        window.trigger_panel.select_entry("effect", PATROL_EFFECT)
        assert window.map_view.area_edit() == _gen().GEOMETRY_PATROL_AREA
        window.area_tile_borders_check.setChecked(False)
        assert settings.get_tool_option_default("area_tile_borders") is True
    finally:
        conftest.close_window(window)


def test_a_read_only_file_disables_the_box_and_its_handles() -> None:
    window = _editing()
    try:
        panel = window.trigger_panel
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        panel.select_trigger(_gen().REFERENCE_TRIGGER)
        panel.select_entry("effect", PATROL_EFFECT)
        window._update_tool_enabled()
        assert not panel._editable
        assert not window.area_edit_check.isEnabled()
        assert window.map_view.area_edit() is None
    finally:
        conftest.close_window(window)


def test_turning_it_on_with_no_area_shown_says_how_to_get_one() -> None:
    hint = "Select one condition or effect with an area or location to edit it on the map"
    window = _window()
    try:
        window.trigger_panel.select_entry("effect", HALF_SET_EFFECT)
        window.area_edit_check.setChecked(True)
        assert _last_status(window) == hint
        window.area_edit_check.setChecked(False)
        window.trigger_panel.select_entry("effect", PATROL_EFFECT)
        window.area_edit_check.setChecked(True)
        assert window.status_log.toPlainText().splitlines().count(hint) == 1, "an editable area: no hint"
    finally:
        conftest.close_window(window)


def test_with_the_box_off_a_drag_on_the_area_pans_and_writes_nothing() -> None:
    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry("effect", PATROL_EFFECT)
        assert view.area_edit() is None
        records = _records(window)
        _move(view, _point(view, "move"))
        assert view._area_hover is None
        _drag(view, _point(view, "move"), _ground_pos(view, 50.5, 50.5))
        assert _records(window) == records
        assert _values(window, "effect", PATROL_EFFECT, AREA) == _gen().GEOMETRY_PATROL_AREA
    finally:
        conftest.close_window(window)


def test_a_drag_off_the_area_pans() -> None:
    from PyQt5.QtWidgets import QGraphicsView

    window = _editing()
    try:
        view = window.map_view
        records = _records(window)
        _move(view, _ground_pos(view, 40.5, 50.5))
        assert view._area_hover is None
        assert view.dragMode() == QGraphicsView.ScrollHandDrag
        _drag(view, _ground_pos(view, 40.5, 50.5), _ground_pos(view, 45.5, 55.5))
        assert view._area_drag is None
        assert _records(window) == records
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("tile_borders", [False, True])
def test_with_resize_off_the_whole_area_moves(tile_borders) -> None:
    window = _editing()
    try:
        view = window.map_view
        window.area_tile_borders_check.setChecked(tile_borders)
        window.area_resize_check.setChecked(False)
        _move(view, _point(view, "w"))
        assert view._area_hover == "move"
        _move(view, _ground_pos(view, 30.85, 43.5))
        assert view._area_hover == "move", "the tile ring does not resize either"
        _drag(view, _point(view, "w"), _ground_pos(view, 27.5, 43.5))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (27, 40, 31, 46)
        assert _last_status(window).endswith("Move Area")
    finally:
        conftest.close_window(window)


def test_with_move_off_only_the_edges_grab() -> None:
    window = _editing()
    try:
        view = window.map_view
        window.area_move_check.setChecked(False)
        records = _records(window)
        _move(view, _point(view, "move"))
        assert view._area_hover is None
        _drag(view, _point(view, "move"), _ground_pos(view, 40.5, 50.5))
        assert _records(window) == records, "the middle pans"
        _drag(view, _point(view, "s"), _ground_pos(view, 32.5, 50.5))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (30, 40, 34, 50)
    finally:
        conftest.close_window(window)


def test_with_both_options_off_nothing_grabs() -> None:
    window = _editing()
    try:
        view = window.map_view
        window.area_move_check.setChecked(False)
        window.area_resize_check.setChecked(False)
        for zone in ("move", "w", "se"):
            _move(view, _point(view, zone))
            assert view._area_hover is None, zone
        _press(view, _point(view, "se"))
        assert view._area_drag is None
        _release(view, _point(view, "se"))
    finally:
        conftest.close_window(window)


def test_tile_borders_greys_out_while_resize_is_off() -> None:
    window = _editing()
    try:
        assert window.area_tile_borders_check.isEnabled()
        window.area_resize_check.setChecked(False)
        assert not window.area_tile_borders_check.isEnabled()
        window.area_resize_check.setChecked(True)
        assert window.area_tile_borders_check.isEnabled()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Flat", "Flat top-down", "Stepped"])
def test_tile_borders_make_the_whole_edge_tile_resize(style) -> None:
    window = _editing(style)
    try:
        view = window.map_view
        deep = _ground_pos(view, 30.85, 43.5)  # inside the west edge tile, past the thin band
        assert view._area_band_tiles() < 0.85, "the thin band must not reach the probe"
        _move(view, deep)
        assert view._area_hover == "move"
        window.area_tile_borders_check.setChecked(True)
        _move(view, deep)
        assert view._area_hover == "w"
        _move(view, _ground_pos(view, 32.5, 43.5))
        assert view._area_hover == "move", "a 5x7 area keeps its middle"
        _move(view, _ground_pos(view, 32.5, 40.85))
        assert view._area_hover == "n"
        _drag(view, deep, _ground_pos(view, 28.5, 43.5))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (28, 40, 34, 46)
    finally:
        conftest.close_window(window)


def test_tile_borders_leave_a_two_tile_axis_thin() -> None:
    gen = _gen()
    window = _editing()
    try:
        view = window.map_view
        window.set_entry_field_group(
            gen.REFERENCE_TRIGGER, ("effect", PATROL_EFFECT), dict(zip(AREA, (30, 40, 31, 46), strict=True)), "Set Area"
        )
        assert view.area_edit() == (30, 40, 31, 46)
        window.area_tile_borders_check.setChecked(True)
        _move(view, _ground_pos(view, 31.0, 43.5))
        assert view._area_hover == "move", "2 tiles wide: no tile border across x"
        _move(view, _ground_pos(view, 31.0, 40.85))
        assert view._area_hover == "n", "7 tiles tall: the end tile is border"
    finally:
        conftest.close_window(window)


def test_unticking_an_option_mid_drag_drops_the_drag() -> None:
    window = _editing()
    try:
        view = window.map_view
        records = _records(window)
        _press(view, _point(view, "move"))
        _move(view, _ground_pos(view, 40.5, 50.5))
        window.area_resize_check.setChecked(False)
        assert view._area_drag is None and view._tile_pick_item is None
        _release(view, _ground_pos(view, 40.5, 50.5))
        assert _records(window) == records
    finally:
        conftest.close_window(window)


# -- dragging ---------------------------------------------------------------------


@pytest.mark.parametrize("style", STYLES)
def test_dragging_the_middle_moves_the_area_and_keeps_its_size(style) -> None:
    window = _editing(style)
    try:
        view = window.map_view
        records = _records(window)
        _press(view, _point(view, "move"))
        _move(view, _ground_pos(view, 37.5, 40.5))
        assert view._tile_pick_item is not None, "the drag previews"
        assert _values(window, "effect", PATROL_EFFECT, AREA) == _gen().GEOMETRY_PATROL_AREA, "nothing until release"
        _release(view, _ground_pos(view, 37.5, 40.5))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (35, 37, 39, 43)
        assert _records(window) == records + 1
        assert _last_status(window).endswith("Move Area")
        assert view._tile_pick_item is None
        assert view.area_edit() == (35, 37, 39, 43), "the handles follow the write"
        window.undo()
        assert _values(window, "effect", PATROL_EFFECT, AREA) == _gen().GEOMETRY_PATROL_AREA
        assert view.area_edit() == _gen().GEOMETRY_PATROL_AREA, "and the undo"
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize(
    ("zone", "end", "expected"),
    [
        ("w", (27.5, 43.5), (27, 40, 34, 46)),
        ("s", (32.5, 50.5), (30, 40, 34, 50)),
        ("ne", (38.5, 38.5), (30, 38, 38, 46)),
        ("se", (20.5, 30.5), (20, 30, 30, 40)),  # crosses both opposite sides
    ],
)
def test_dragging_an_edge_or_corner_resizes_the_area(style, zone, end, expected) -> None:
    window = _editing(style)
    try:
        view = window.map_view
        records = _records(window)
        start = _point(view, zone)
        _press(view, start)
        assert view._area_drag is not None and view._area_drag[0] == zone
        _move(view, _ground_pos(view, *end))
        _release(view, _ground_pos(view, *end))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == expected
        assert _records(window) == records + 1
        assert _last_status(window).endswith("Resize Area")
    finally:
        conftest.close_window(window)


def test_a_drag_past_the_map_edge_moves_the_area_only_to_the_edge() -> None:
    window = _editing()
    try:
        view = window.map_view
        _drag(view, _point(view, "move"), _ground_pos(view, -20.5, 50.5))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (0, 47, 4, 53)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("zone", ["move", "w", "se"])
def test_a_click_with_no_motion_records_nothing(zone) -> None:
    window = _editing()
    try:
        view = window.map_view
        records = _records(window)
        _press(view, _point(view, zone))
        _release(view, _point(view, zone))
        assert _records(window) == records
        assert view._area_drag is None and view._tile_pick_item is None
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", STYLES)
def test_hover_cursors_follow_the_zone_on_screen(style) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QGraphicsView

    window = _editing(style)
    try:
        view = window.map_view
        top_down = style == "Flat top-down"
        expected = {
            "move": Qt.SizeAllCursor,
            # Top-down: map x is screen x. Iso: map x runs up-right, map y down-right.
            "w": Qt.SizeHorCursor if top_down else Qt.SizeBDiagCursor,
            "n": Qt.SizeVerCursor if top_down else Qt.SizeFDiagCursor,
            "nw": Qt.SizeFDiagCursor if top_down else Qt.SizeHorCursor,
            "ne": Qt.SizeBDiagCursor if top_down else Qt.SizeVerCursor,
        }
        for zone, shape in expected.items():
            _move(view, _point(view, zone))
            assert view._area_hover == zone
            assert view.cursor().shape() == shape, zone
            assert view.dragMode() == QGraphicsView.NoDrag
        _move(view, _ground_pos(view, 60.5, 60.5))
        assert view._area_hover is None
        assert view.dragMode() == QGraphicsView.ScrollHandDrag, "back to the Pan hand"
    finally:
        conftest.close_window(window)


# -- cancels ----------------------------------------------------------------------


@pytest.mark.parametrize("grab", ["area", "location"])
@pytest.mark.parametrize("how", ["escape", "right_click", "restyle", "mode", "untick", "ruler"])
def test_every_cancel_drops_the_drag_without_writing(how, grab) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _editing("Stepped")
    try:
        view = window.map_view
        records = _records(window)
        _press(view, _point(view, "move") if grab == "area" else _mark_pos(view))
        _move(view, _ground_pos(view, 40.5, 50.5))
        if how == "escape":
            _escape(view)
        elif how == "right_click":
            _press(view, _ground_pos(view, 40.5, 50.5), Qt.RightButton)
        elif how == "restyle":
            window.terrain_style_combo.setCurrentText("Sloped")
            window.refresh_map()
            QApplication.processEvents()
        elif how == "mode":
            window.mode_combo.setCurrentText("View")
        elif how == "untick":
            window.area_edit_check.setChecked(False)
        else:
            window.ruler_action.setChecked(True)
        assert view._area_drag is None
        assert view._tile_pick_item is None and view._tile_pick_label is None, "no preview left behind"
        _move(view, _ground_pos(view, 45.5, 55.5))
        _release(view, _ground_pos(view, 45.5, 55.5))
        assert _records(window) == records
        assert _values(window, "effect", PATROL_EFFECT, AREA) == _gen().GEOMETRY_PATROL_AREA
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == _gen().GEOMETRY_PATROL_LOCATION
    finally:
        conftest.close_window(window)


def test_clear_image_forgets_a_live_drag() -> None:
    window = _editing()
    try:
        view = window.map_view
        _press(view, _point(view, "move"))
        view.clear_image()
        assert view._area_drag is None and view._tile_pick_item is None
    finally:
        conftest.close_window(window)


# -- when there are no handles ----------------------------------------------------


def test_no_handles_for_several_entries_a_half_set_area_or_an_area_free_entry() -> None:
    window = _editing()
    try:
        panel = window.trigger_panel
        panel.select_entries([("effect", PATROL_EFFECT), ("effect", 5)])
        assert window.map_view.area_edit() is None
        panel.select_entry("effect", HALF_SET_EFFECT)
        assert window.map_view.area_edit() is None
        panel.select_entry("effect", CREATE_EFFECT)
        assert window.map_view.area_edit() is None
        panel.select_entry("effect", PATROL_EFFECT)
        assert window.map_view.area_edit() == _gen().GEOMETRY_PATROL_AREA
    finally:
        conftest.close_window(window)


def test_no_handles_with_the_overlay_off_or_another_tool() -> None:
    window = _editing()
    try:
        window.trigger_overlay_action.setChecked(False)
        assert window.map_view.area_edit() is None
        window.trigger_overlay_action.setChecked(True)
        assert window.map_view.area_edit() == _gen().GEOMETRY_PATROL_AREA
        window.ruler_action.setChecked(True)
        assert window.map_view.area_edit() is None
        window.pan_action.setChecked(True)
        assert window.map_view.area_edit() == _gen().GEOMETRY_PATROL_AREA
    finally:
        conftest.close_window(window)


def test_an_armed_set_pick_wins_over_the_handles() -> None:
    window = _editing()
    try:
        view = window.map_view
        records = _records(window)
        _arm(window, "area")
        _move(view, _point(view, "move"))
        assert view._area_hover is None
        _press(view, _point(view, "move"))
        assert view._area_drag is None and view._tile_pick_anchor is not None, "the press anchors the pick"
        _release(view, _ground_pos(view, 50.5, 50.5))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (32, 43, 50, 50)
        assert _records(window) == records + 1
        assert _last_status(window).endswith("Set Area")
        assert view.area_edit() == (32, 43, 50, 50), "disarmed, the handles come back on the new area"
    finally:
        conftest.close_window(window)


def test_a_drag_whose_entry_stops_being_shown_writes_nothing() -> None:
    window = _editing()
    try:
        view = window.map_view
        records = _records(window)
        _press(view, _point(view, "move"))
        window.trigger_panel.select_entry("effect", CREATE_EFFECT)
        assert view._area_drag is None
        _release(view, _ground_pos(view, 50.5, 50.5))
        # The callback itself also refuses an entry the form no longer shows.
        window._area_edit_target = (_gen().REFERENCE_TRIGGER, "effect", PATROL_EFFECT, _gen().GEOMETRY_PATROL_AREA)
        view.on_area_edit(1, 2, 3, 4)
        assert _records(window) == records
        assert _values(window, "effect", PATROL_EFFECT, AREA) == _gen().GEOMETRY_PATROL_AREA
    finally:
        conftest.close_window(window)


def test_a_drag_whose_release_was_swallowed_dies_at_the_next_buttonless_move() -> None:
    from PyQt5.QtCore import QEvent, Qt

    window = _editing()
    try:
        view = window.map_view
        records = _records(window)
        _press(view, _point(view, "move"))
        assert view._area_drag is not None
        end = _ground_pos(view, 50.5, 50.5)
        view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.NoButton))
        assert view._area_drag is None
        _release(view, end)
        assert _records(window) == records
        assert _values(window, "effect", PATROL_EFFECT, AREA) == _gen().GEOMETRY_PATROL_AREA
    finally:
        conftest.close_window(window)


def test_a_drag_whose_entry_changes_to_one_with_the_same_area_writes_nothing() -> None:
    window = _editing()
    try:
        view = window.map_view
        area = _gen().GEOMETRY_PATROL_AREA
        window.set_entry_field_group(
            _gen().REFERENCE_TRIGGER, ("effect", HALF_SET_EFFECT), dict(zip(AREA, area, strict=True)), "Set Area"
        )
        window.trigger_panel.select_entry("effect", PATROL_EFFECT)
        assert view.area_edit() == area
        records = _records(window)
        _press(view, _point(view, "move"))
        window.trigger_panel.select_entry("effect", HALF_SET_EFFECT)
        assert view.area_edit() == area
        assert view._area_drag is None
        _move(view, _ground_pos(view, 50.5, 50.5))
        _release(view, _ground_pos(view, 50.5, 50.5))
        assert _records(window) == records
        assert _values(window, "effect", HALF_SET_EFFECT, AREA) == area
    finally:
        conftest.close_window(window)


# -- GH #41: dragging the location mark ------------------------------------------------


@pytest.mark.parametrize("style", STYLES)
def test_dragging_the_mark_moves_the_location_and_leaves_the_area(style) -> None:
    gen = _gen()
    window = _editing(style)
    try:
        view = window.map_view
        assert view.location_edit() == gen.GEOMETRY_PATROL_LOCATION
        records = _records(window)
        _press(view, _mark_pos(view))
        assert view._area_drag is not None and view._area_drag[0] == "location"
        _move(view, _mark_pos(view, (57, 44)))
        assert view._tile_pick_item is not None, "the drag previews"
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == gen.GEOMETRY_PATROL_LOCATION, "nothing until release"
        _release(view, _mark_pos(view, (57, 44)))
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == (57, 44)
        assert _values(window, "effect", PATROL_EFFECT, AREA) == gen.GEOMETRY_PATROL_AREA
        assert _records(window) == records + 1
        assert _last_status(window).endswith("Move Location")
        assert view._tile_pick_item is None
        assert view.location_edit() == (57, 44), "the handle follows the write"
        window.undo()
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == gen.GEOMETRY_PATROL_LOCATION
        assert view.location_edit() == gen.GEOMETRY_PATROL_LOCATION, "and the undo"
    finally:
        conftest.close_window(window)


def test_a_location_only_entry_drags_its_mark_with_no_area() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QGraphicsView

    gen = _gen()
    window = _editing()
    try:
        view = window.map_view
        window.trigger_panel.select_entry("effect", CREATE_EFFECT)
        assert view.area_edit() is None
        assert view.location_edit() == gen.GEOMETRY_CREATE_LOCATION
        mark = _mark_pos(view, gen.GEOMETRY_CREATE_LOCATION)
        _move(view, mark)
        assert view._area_hover == "location"
        assert view.cursor().shape() == Qt.SizeAllCursor
        assert view.dragMode() == QGraphicsView.NoDrag
        _drag(view, mark, _mark_pos(view, (72, 83)))
        assert _values(window, "effect", CREATE_EFFECT, LOCATION) == (72, 83)
        assert _last_status(window).endswith("Move Location")
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("entry", [PATROL_EFFECT, CREATE_EFFECT])
@pytest.mark.parametrize(
    ("move", "resize", "grabs"), [(True, True, True), (True, False, True), (False, True, False), (False, False, False)]
)
def test_only_move_gates_the_mark(entry, move, resize, grabs) -> None:
    from PyQt5.QtWidgets import QGraphicsView

    window = _editing()
    try:
        view = window.map_view
        window.trigger_panel.select_entry("effect", entry)
        window.area_move_check.setChecked(move)
        window.area_resize_check.setChecked(resize)
        x, y = view.location_edit()
        records = _records(window)
        _move(view, _mark_pos(view, (x, y)))
        assert view._area_hover == ("location" if grabs else None)
        assert view.dragMode() == (QGraphicsView.NoDrag if grabs else QGraphicsView.ScrollHandDrag)
        _drag(view, _mark_pos(view, (x, y)), _mark_pos(view, (x + 3, y + 2)))
        assert _records(window) == records + (1 if grabs else 0), "else the drag pans"
        assert _values(window, "effect", entry, LOCATION) == ((x + 3, y + 2) if grabs else (x, y))
    finally:
        conftest.close_window(window)


def test_a_mark_inside_the_area_wins_and_beside_it_the_area_still_moves() -> None:
    gen = _gen()
    window = _editing()
    try:
        view = window.map_view
        _set_location(window, PATROL_EFFECT, (32, 43))
        _move(view, _mark_pos(view, (32, 43)))
        assert view._area_hover == "location"
        _move(view, _ground_pos(view, 33.5, 43.5))
        assert view._area_hover == "move"
        _drag(view, _mark_pos(view, (32, 43)), _mark_pos(view, (36, 48)))
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == (36, 48)
        assert _values(window, "effect", PATROL_EFFECT, AREA) == gen.GEOMETRY_PATROL_AREA
    finally:
        conftest.close_window(window)


def test_a_one_tile_area_under_the_mark_keeps_its_rim() -> None:
    """Decisions 3 and 4: the mark takes the middle, the rim still resizes, and Move off gives the middle back."""
    gen = _gen()
    window = _editing()
    try:
        view = window.map_view
        window.set_entry_field_group(
            gen.REFERENCE_TRIGGER, ("effect", PATROL_EFFECT), dict(zip(AREA, (32, 43, 32, 43), strict=True)), "Set Area"
        )
        _set_location(window, PATROL_EFFECT, (32, 43))
        view.scale(4.0, 4.0)
        assert view._area_band_tiles() < 0.4, "the rim probe must sit past the reach"
        _move(view, _mark_pos(view, (32, 43)))
        assert view._area_hover == "location"
        _move(view, _ground_pos(view, 32.05, 43.5))
        assert view._area_hover == "w"
        window.area_move_check.setChecked(False)
        _move(view, _mark_pos(view, (32, 43)))
        assert view._area_hover is None
        _move(view, _ground_pos(view, 32.05, 43.5))
        assert view._area_hover == "w"
    finally:
        conftest.close_window(window)


def test_a_location_its_object_overrides_has_no_handle() -> None:
    gen = _gen()
    window = _editing()
    try:
        view = window.map_view
        window.trigger_panel.select_entry("effect", gen.TASK_OBJECT_EFFECT)
        assert view.location_edit() is None
        records = _records(window)
        mark = _mark_pos(view, gen.HOUSE_TILE)
        _move(view, mark)
        assert view._area_hover is None
        _press(view, mark)
        assert view._area_drag is None, "the drag pans"
        _move(view, _mark_pos(view, (45, 60)))
        _release(view, _mark_pos(view, (45, 60)))
        assert _records(window) == records
        assert _values(window, "effect", gen.TASK_OBJECT_EFFECT, LOCATION) == gen.HOUSE_TILE
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_mark_grabs_at_its_raised_tile(style) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    gen = _gen()
    tile = gen.GEOMETRY_PATROL_LOCATION
    window = _editing(style)
    try:
        view = window.map_view
        before = view._tile_polygon(*tile).boundingRect().center().y()
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(*tile, Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        assert window.scenario.map_manager.get_tile(*tile).elevation > 0
        window.mode_combo.setCurrentText("Triggers")
        window._on_tool_selected("pan")
        QApplication.processEvents()
        assert view.location_edit() == tile
        assert view._tile_polygon(*tile).boundingRect().center().y() < before, "the tile must really rise"
        _press(view, conftest.polygon_viewport_pos(view, *tile))
        assert view._area_drag is not None and view._area_drag[0] == "location"
        _move(view, _mark_pos(view, (55, 45)))
        _release(view, _mark_pos(view, (55, 45)))
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == (55, 45)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(("end", "expected"), [((-20.5, 50.5), (0, 50)), ((140.5, 130.5), (119, 119))])
def test_a_drag_past_the_map_edge_stops_the_mark_on_the_edge_tile(end, expected) -> None:
    window = _editing()
    try:
        view = window.map_view
        _drag(view, _mark_pos(view), _ground_pos(view, *end))
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == expected
    finally:
        conftest.close_window(window)


def test_a_click_on_the_mark_records_nothing() -> None:
    window = _editing()
    try:
        view = window.map_view
        records = _records(window)
        _press(view, _mark_pos(view))
        _release(view, _mark_pos(view))
        assert _records(window) == records
        assert view._area_drag is None and view._tile_pick_item is None and view._tile_pick_label is None
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", STYLES)
def test_the_preview_shows_the_mark_the_line_and_the_rulers_distance(style) -> None:
    from descape import ruler, trigger_geometry

    gen = _gen()
    origin = trigger_geometry.area_centre(gen.GEOMETRY_PATROL_AREA)
    window = _editing(style)
    try:
        view = window.map_view
        (old_label,) = [i for i in view.trigger_overlay_items() if i.data(0) == "label"]
        old_text = old_label.text()
        _press(view, _mark_pos(view))
        label = view._tile_pick_label
        for tile in ((57, 44), (58, 60)):
            _move(view, _mark_pos(view, tile))
            assert view._tile_pick_label is label, "reused across moves"
            assert label.text() == ruler.format_measurement(ruler.measure(origin, tile))
            assert label.pos() == view._ruler_anchor(tile)
            box = view._tile_pick_item.path().boundingRect()
            assert box.contains(view._location_mark(tile).boundingRect())
            assert box.contains(view._ruler_anchor(origin))
        assert old_label.text() == old_text, "the old run stays until release"
        _release(view, _mark_pos(view, (58, 60)))
        assert view._tile_pick_item is None and view._tile_pick_label is None
    finally:
        conftest.close_window(window)


def test_a_selection_run_previews_from_the_units_centroid() -> None:
    from descape import ruler

    gen = _gen()
    window = _editing()
    try:
        view = window.map_view
        _set_location(window, gen.PATROL_IDS_EFFECT, (70, 70))
        window.trigger_panel.select_entry("effect", gen.PATROL_IDS_EFFECT)
        assert view.area_edit() is None
        _press(view, _mark_pos(view, (70, 70)))
        _move(view, _mark_pos(view, (72, 73)))
        origin = (int(gen.ARCHER_B_POS[0]), int(gen.ARCHER_B_POS[1]))
        assert view._tile_pick_label.text() == ruler.format_measurement(ruler.measure(origin, (72, 73)))
    finally:
        conftest.close_window(window)


def test_no_run_origin_on_the_map_previews_the_mark_alone() -> None:
    gen = _gen()
    window = _editing()
    try:
        view = window.map_view
        window.trigger_panel.select_entry("effect", CREATE_EFFECT)
        _press(view, _mark_pos(view, gen.GEOMETRY_CREATE_LOCATION))
        _move(view, _mark_pos(view, (72, 83)))
        assert view._tile_pick_label is None
        box = view._tile_pick_item.path().boundingRect()
        assert box == view._location_mark((72, 83)).boundingRect(), "no line either"
        _escape(view)
        view.set_location_edit(gen.GEOMETRY_CREATE_LOCATION, (500, 500))
        _press(view, _mark_pos(view, gen.GEOMETRY_CREATE_LOCATION))
        _move(view, _mark_pos(view, (72, 83)))
        assert view._tile_pick_label is None
    finally:
        conftest.close_window(window)


def test_clear_image_forgets_a_live_mark_drag() -> None:
    window = _editing()
    try:
        view = window.map_view
        _press(view, _mark_pos(view))
        assert view._tile_pick_label is not None
        view.clear_image()
        assert view._area_drag is None
        assert view._tile_pick_item is None and view._tile_pick_label is None
        view.set_tile_picker(None)  # would removeItem() a deleted item
    finally:
        conftest.close_window(window)


def test_no_mark_handle_past_the_map_with_the_box_or_overlay_off_or_for_several_entries() -> None:
    gen = _gen()
    window = _editing()
    try:
        view = window.map_view
        panel = window.trigger_panel
        location = gen.GEOMETRY_PATROL_LOCATION
        window.trigger_overlay_action.setChecked(False)
        assert view.location_edit() is None
        window.trigger_overlay_action.setChecked(True)
        assert view.location_edit() == location
        window.ruler_action.setChecked(True)
        assert view.location_edit() is None
        window.pan_action.setChecked(True)
        assert view.location_edit() == location
        panel.select_entries([("effect", PATROL_EFFECT), ("effect", CREATE_EFFECT)])
        assert view.location_edit() is None
        panel.select_entry("effect", PATROL_EFFECT)
        assert view.location_edit() == location
        window.area_edit_check.setChecked(False)
        assert view.location_edit() is None
        window.area_edit_check.setChecked(True)
        _set_location(window, PATROL_EFFECT, (500, 50))
        assert view.location_edit() is None
    finally:
        conftest.close_window(window)


def test_an_armed_set_location_wins_over_the_handle() -> None:
    window = _editing()
    try:
        view = window.map_view
        _arm(window, "location")
        _move(view, _mark_pos(view))
        assert view._area_hover is None
        _press(view, _mark_pos(view))
        assert view._area_drag is None, "the press is the pick's"
        assert view.tile_picker_kind() is None
        assert view.location_edit() == _gen().GEOMETRY_PATROL_LOCATION, "disarmed, the handle comes back"
    finally:
        conftest.close_window(window)


def test_the_location_callback_refuses_an_entry_the_form_no_longer_shows() -> None:
    gen = _gen()
    window = _editing()
    try:
        records = _records(window)
        window.trigger_panel.select_entry("effect", CREATE_EFFECT)
        window._location_edit_target = (gen.REFERENCE_TRIGGER, "effect", PATROL_EFFECT, gen.GEOMETRY_PATROL_LOCATION)
        window.map_view.on_location_edit(1, 2)
        assert _records(window) == records
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == gen.GEOMETRY_PATROL_LOCATION
    finally:
        conftest.close_window(window)


def test_a_mark_drag_whose_entry_changes_to_one_with_the_same_coords_writes_nothing() -> None:
    gen = _gen()
    window = _editing()
    try:
        view = window.map_view
        area, location = gen.GEOMETRY_PATROL_AREA, gen.GEOMETRY_PATROL_LOCATION
        twin = ("effect", gen.PATROL_IDS_EFFECT)
        window.set_entry_field_group(gen.REFERENCE_TRIGGER, twin, dict(zip(AREA, area, strict=True)), "Set Area")
        _set_location(window, gen.PATROL_IDS_EFFECT, location)
        window.trigger_panel.select_entry("effect", PATROL_EFFECT)
        assert (view.area_edit(), view.location_edit()) == (area, location)
        records = _records(window)
        _press(view, _mark_pos(view))
        window.trigger_panel.select_entry(*twin)
        assert (view.area_edit(), view.location_edit()) == (area, location)
        assert view._area_drag is None
        _move(view, _mark_pos(view, (55, 45)))
        _release(view, _mark_pos(view, (55, 45)))
        assert _records(window) == records
        assert _values(window, *twin, LOCATION) == location
    finally:
        conftest.close_window(window)
