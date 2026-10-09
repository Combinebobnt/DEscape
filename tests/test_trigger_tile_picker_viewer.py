"""GH #138: Set / Go to / Reset rows for a condition or effect's Location,
Area and Objects groups, through a real offscreen ViewerWindow on
tests/fixtures/triggers_120x120.aoe2scenario: the group write funnel, the
rows, the map's point and rectangle picks, and Go to."""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
MAP_SIZE = 120
# Trigger 3's entries (tools/gen_trigger_fixture.py).
AREA_CONDITION = 1
PATROL_EFFECT = 1
CREATE_EFFECT = 2
HALF_SET_EFFECT = 3
AREA = ("area_x1", "area_y1", "area_x2", "area_y2")
LOCATION = ("location_x", "location_y")


def _gen():
    return conftest.load_verify_module("gen_trigger_fixture")


# Combo text -> the MapView style it renders as. "Flat" is Stepped with zero
# elevations; "Flat top-down" is the real Flat, whose _pick_tile has no bounds check.
MAP_STYLE = {"Flat": "stepped", "Flat top-down": "flat", "Stepped": "stepped", "Sloped": "sloped"}


def _window(style: str = "Flat"):
    """Re-rendered through refresh_map(): offscreen, a restyle alone never
    reaches MapView (GOTCHAS, Tests), so the style each case ran is asserted."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1100, 760)
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None
    window.terrain_style_combo.setCurrentText(style.split(maxsplit=1)[0])
    if style.startswith("Flat"):
        window.iso_action.setChecked(style == "Flat")
    window.refresh_map()
    QApplication.processEvents()
    assert window.map_view._terrain_style == MAP_STYLE[style]
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_trigger(_gen().REFERENCE_TRIGGER)
    return window


def _entry(window, kind, index):
    from descape.scenario_io import parse_triggers

    model = window.trigger_edits
    manager = model.manager() if model is not None else parse_triggers(window.scenario)
    trigger = manager.triggers[_gen().REFERENCE_TRIGGER]
    return (trigger.conditions if kind == "condition" else trigger.effects)[index]


def _values(window, kind, index, fields):
    entry = _entry(window, kind, index)
    return tuple(getattr(entry, f) for f in fields)


def _records(window) -> int:
    return len(window.edit_history.records)


def _row(panel, field):
    for spec, _kind, _index, widget in panel._rows:
        if spec.name == field:
            return widget
    raise AssertionError(f"no {field} row in {[s.name for s, *_ in panel._rows]}")


# -- the group write funnel -----------------------------------------------------


def test_a_group_write_is_one_record_and_one_undo_restores_every_field() -> None:
    gen = _gen()
    window = _window()
    try:
        window.trigger_panel.select_entry("effect", PATROL_EFFECT)
        records = _records(window)
        values = dict(zip(AREA, (3, 4, 9, 12), strict=True))
        window.set_entry_field_group(gen.REFERENCE_TRIGGER, ("effect", PATROL_EFFECT), values, "Set Area")
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (3, 4, 9, 12)
        assert _records(window) == records + 1
        assert window.status_log.toPlainText().splitlines()[-1].endswith("Set Area")
        window.undo()
        assert _values(window, "effect", PATROL_EFFECT, AREA) == gen.GEOMETRY_PATROL_AREA
    finally:
        conftest.close_window(window)


def test_a_group_write_that_changes_nothing_records_nothing() -> None:
    gen = _gen()
    window = _window()
    try:
        records = _records(window)
        values = dict(zip(LOCATION, gen.GEOMETRY_PATROL_LOCATION, strict=True))
        window.set_entry_field_group(gen.REFERENCE_TRIGGER, ("effect", PATROL_EFFECT), values, "Set Location")
        assert _records(window) == records
    finally:
        conftest.close_window(window)


def test_a_group_write_repopulates_the_form_so_the_spinboxes_show_it() -> None:
    """A field edit only patches labels (set_entry_fields), so the group write
    repopulates itself or the spinboxes keep showing the old values."""
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", PATROL_EFFECT)
        values = dict(zip(LOCATION, (7, 8), strict=True))
        window.set_entry_field_group(gen.REFERENCE_TRIGGER, ("effect", PATROL_EFFECT), values, "Set Location")
        assert (_row(panel, "location_x").value(), _row(panel, "location_y").value()) == (7, 8)
    finally:
        conftest.close_window(window)


# -- the rows -------------------------------------------------------------------


def _group(panel, group):
    row = panel._group_rows.get(group)
    assert row is not None, f"no {group} row in {sorted(panel._group_rows)}"
    return row


def _form_row(panel, widget) -> int:
    row, _role = panel.property_form.getWidgetPosition(widget)
    assert row >= 0
    return row


@pytest.mark.parametrize(
    ("kind", "index", "groups"),
    [
        ("effect", PATROL_EFFECT, {"location", "area", "objects"}),
        ("effect", CREATE_EFFECT, {"location"}),
        ("condition", AREA_CONDITION, {"area"}),
        ("condition", 2, set()),  # destroy_object: a Unit field, no group
        ("effect", 5, {"location", "area", "objects"}),  # task_object
    ],
)
def test_rows_appear_only_for_the_groups_the_definition_carries(kind, index, groups) -> None:
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry(kind, index)
        assert set(panel._group_rows) == groups
        for group, row in panel._group_rows.items():
            assert [b.text() for b in (row.set_button, row.go_button, row.reset_button)] == ["Set", "Go to", "Reset"]
            caption = panel.property_form.labelForField(row.set_button.parentWidget())
            assert caption is not None and caption.text() == group.capitalize()
            # Right after the group's last field row.
            last = {"location": "location_y", "area": "area_y2", "objects": "selected_object_ids"}[group]
            assert _form_row(panel, row.set_button.parentWidget()) == _form_row(panel, _row(panel, last)) + 1
        hosts = {row.set_button.parentWidget() for row in panel._group_rows.values()}
        assert not any(widget in hosts for *_, widget in panel._rows), "group rows stay out of _rows"
    finally:
        conftest.close_window(window)


def test_task_object_puts_the_location_row_before_the_location_object_row() -> None:
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", 5)
        location = _form_row(panel, _group(panel, "location").set_button.parentWidget())
        assert location < _form_row(panel, _row(panel, "location_object_reference"))
    finally:
        conftest.close_window(window)


def test_a_multi_entry_selection_gets_no_group_rows() -> None:
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entries([("effect", PATROL_EFFECT), ("effect", 5)])
        assert len(panel._form_refs) == 2
        assert panel._group_rows == {}
    finally:
        conftest.close_window(window)


def test_set_objects_is_the_unit_picker_relabelled_into_the_objects_row() -> None:
    from PyQt5.QtWidgets import QToolButton

    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", 5)
        target = (gen.REFERENCE_TRIGGER, "effect", 5, "selected_object_ids", True)
        assert panel._pick_buttons[target] is _group(panel, "objects").set_button
        host = _row(panel, "selected_object_ids")
        assert [b.text() for b in host.findChildren(QToolButton)] == [], "the old Pick button is gone"
        # The scalar Unit field keeps its own Pick.
        assert [b.text() for b in _row(panel, "location_object_reference").findChildren(QToolButton)] == ["Pick"]
        _group(panel, "objects").set_button.click()
        assert window._unit_picker == target
        window.map_view.on_unit_pick((1, gen.REF_HOUSE))
        assert list(_values(window, "effect", 5, ("selected_object_ids",))[0]) == [*gen.TASK_SELECTED_IDS, gen.REF_HOUSE]
        assert window._unit_picker == target, "a list pick stays armed"
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(
    ("kind", "index", "group", "go", "reset"),
    [
        ("effect", PATROL_EFFECT, "area", True, True),
        ("effect", PATROL_EFFECT, "location", True, True),
        ("effect", HALF_SET_EFFECT, "area", False, True),  # (-1, 5, 12, 9): draws nothing, still resettable
        ("effect", 6, "objects", True, True),  # patrol with one placed archer
        ("effect", 6, "location", False, False),  # never set
    ],
)
def test_go_to_and_reset_gate_on_the_stored_values(kind, index, group, go, reset) -> None:
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry(kind, index)
        row = _group(panel, group)
        assert row.set_button.isEnabled()
        assert (row.go_button.isEnabled(), row.reset_button.isEnabled()) == (go, reset)
    finally:
        conftest.close_window(window)


def test_go_to_objects_needs_an_id_that_resolves_to_a_placed_unit() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        window.set_entry_field_group(
            gen.REFERENCE_TRIGGER, ("effect", 6), {"selected_object_ids": [gen.DANGLING_REFERENCE_ID]}, "Set Objects"
        )
        panel.select_entry("effect", 6)
        row = _group(panel, "objects")
        assert not row.go_button.isEnabled()
        assert row.reset_button.isEnabled()
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_group_rows_do_not_overlap_or_squeeze() -> None:
    """At the 340 px pane the three buttons wrap below their caption at the
    pinned font (WrapLongRows), which is the row class GOTCHAS says passes
    every widget assertion while squeezed. Walked at every swept DPI."""
    from test_trigger_panel import _assert_form_rows_do_not_overlap

    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", PATROL_EFFECT)
        assert len(panel._group_rows) == 3
        walked, _wrapped = _assert_form_rows_do_not_overlap(panel.property_form)
        assert walked >= len(panel._rows) + 3
        assert not panel.property_area.horizontalScrollBar().isVisible()
    finally:
        conftest.close_window(window)


def test_typing_in_a_group_spinbox_regates_go_to_and_reset() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", CREATE_EFFECT)
        row = _group(panel, "location")
        assert row.go_button.isEnabled()
        _row(panel, "location_x").setValue(-1)
        assert _values(window, "effect", CREATE_EFFECT, LOCATION) == (-1, gen.GEOMETRY_CREATE_LOCATION[1])
        assert not row.go_button.isEnabled() and row.reset_button.isEnabled()
        _row(panel, "location_y").setValue(-1)
        assert not row.reset_button.isEnabled()
    finally:
        conftest.close_window(window)


def test_a_read_only_file_keeps_go_to_and_disables_set_and_reset() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        panel.select_trigger(gen.REFERENCE_TRIGGER)
        panel.select_entry("effect", PATROL_EFFECT)
        assert not panel._editable
        for group in ("location", "area", "objects"):
            row = _group(panel, group)
            assert not row.set_button.isEnabled() and not row.reset_button.isEnabled(), group
            # This patrol selects no objects, so only its two coordinate groups can go anywhere.
            assert row.go_button.isEnabled() == (group != "objects"), group
        window.map_view.center_on_tile(5, 5)
        _group(panel, "location").go_button.click()
        assert window.map_view.viewport_centre_tile() == gen.GEOMETRY_PATROL_LOCATION
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(
    ("index", "group", "fields", "unset"),
    [
        (PATROL_EFFECT, "area", AREA, (-1, -1, -1, -1)),
        (PATROL_EFFECT, "location", LOCATION, (-1, -1)),
        (5, "objects", ("selected_object_ids",), ([],)),
    ],
)
def test_reset_clears_the_group_as_one_record_and_greys_out(index, group, fields, unset) -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", index)
        before = _values(window, "effect", index, fields)
        records = _records(window)
        button = _group(panel, group).reset_button
        # Connected after the panel's own slot: the clicked button must still be in the form then.
        parented = []
        button.clicked.connect(lambda _c=False: parented.append(panel.property_host.isAncestorOf(button)))
        button.click()
        assert parented == [True], "the repopulate unparented the button inside its own click"
        QApplication.processEvents()  # the repopulate is deferred past the click
        assert tuple(list(v) if isinstance(v, list) else v for v in _values(window, "effect", index, fields)) == unset
        assert _records(window) == records + 1
        row = _group(panel, group)
        assert not row.reset_button.isEnabled() and not row.go_button.isEnabled()
        if group != "objects":
            assert all(_row(panel, f).value() == -1 for f in fields)
        row.reset_button.click()
        QApplication.processEvents()
        assert _records(window) == records + 1, "a disabled Reset records nothing"
        window.undo()
        assert _values(window, "effect", index, fields) == before
    finally:
        conftest.close_window(window)


# -- the map picks ------------------------------------------------------------------


def _press(view, pos, button=None):
    from PyQt5.QtCore import QEvent, Qt

    button = Qt.LeftButton if button is None else button
    view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, button, button))


def _move(view, pos):
    from PyQt5.QtCore import QEvent, Qt

    view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, pos, Qt.NoButton, Qt.LeftButton))


def _release(view, pos):
    from PyQt5.QtCore import QEvent, Qt

    view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))


def _escape(view):
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))


def _tile_pos(view, tile):
    return conftest.polygon_viewport_pos(view, *tile)


def _ground_pos(view, mx: float, my: float):
    """Viewport position of continuous map point (mx, my) on the elevation-0
    plane: ground_map_point()'s inverse, for points past the map edge."""
    from PyQt5.QtCore import QPointF

    if view._terrain_style == "flat":
        scene = QPointF(mx * view._tile_pixels, my * view._tile_pixels)
    else:
        proj = view._iso_proj
        a, b = (mx - 0.5) + (my - 0.5), (my - 0.5) - (mx - 0.5)
        scene = QPointF(a * proj.half_w + proj.origin_x + proj.half_w, b * proj.half_h + proj.origin_y + proj.half_h)
    point = view.ground_map_point(scene)
    assert (int(point[0]), int(point[1])) == (int(mx), int(my)), "the inverse is wrong"
    return QPointF(view.mapFromScene(scene))


def _arm(window, group, index=PATROL_EFFECT):
    panel = window.trigger_panel
    panel.select_entry("effect", index)
    row = _group(panel, group)
    row.set_button.click()
    assert row.set_button.isChecked()
    assert window._tile_picker is not None and window._tile_picker[3] == group
    return row


STYLES = ["Flat top-down", "Stepped", "Sloped"]


@pytest.mark.parametrize("style", STYLES)
def test_set_location_writes_the_clicked_tile_once_and_disarms(style) -> None:
    window = _window(style)
    try:
        view = window.map_view
        _arm(window, "location")
        assert view.tile_picker_kind() == "point"
        assert window.status_log.toPlainText().splitlines()[-1] == "Click a tile to set the location; Esc to cancel"
        records = _records(window)
        _press(view, _tile_pos(view, (33, 44)))
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == (33, 44)
        assert _records(window) == records + 1
        assert window._tile_picker is None and view.tile_picker_kind() is None
        panel = window.trigger_panel
        assert not _group(panel, "location").set_button.isChecked()
        assert (_row(panel, "location_x").value(), _row(panel, "location_y").value()) == (33, 44)
        window.undo()
        assert _values(window, "effect", PATROL_EFFECT, LOCATION) == _gen().GEOMETRY_PATROL_LOCATION
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", STYLES)
def test_set_area_stores_a_reversed_drag_as_min_and_max(style) -> None:
    window = _window(style)
    try:
        view = window.map_view
        _arm(window, "area")
        assert view.tile_picker_kind() == "rect"
        records = _records(window)
        _press(view, _tile_pos(view, (40, 50)))
        _move(view, _tile_pos(view, (30, 45)))
        _move(view, _tile_pos(view, (20, 30)))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == _gen().GEOMETRY_PATROL_AREA, "nothing until release"
        _release(view, _tile_pos(view, (20, 30)))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (20, 30, 40, 50)
        assert _records(window) == records + 1
        assert window._tile_picker is None and view._tile_pick_item is None
        window.undo()
        assert _values(window, "effect", PATROL_EFFECT, AREA) == _gen().GEOMETRY_PATROL_AREA
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Flat top-down", "Stepped"])
@pytest.mark.parametrize("group", ["location", "area"])
def test_a_press_off_the_map_picks_nothing_and_stays_armed(style, group) -> None:
    window = _window(style)
    try:
        view = window.map_view
        before = _values(window, "effect", PATROL_EFFECT, AREA + LOCATION)
        records = _records(window)
        _arm(window, group)
        off = _ground_pos(view, -5.5, 50.5)
        _press(view, off)
        _release(view, off)
        assert _records(window) == records
        assert _values(window, "effect", PATROL_EFFECT, AREA + LOCATION) == before
        assert window._tile_picker is not None and view.tile_picker_kind() is not None
    finally:
        conftest.close_window(window)


def test_picking_an_object_into_an_empty_list_enables_go_to_objects() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", PATROL_EFFECT)
        row = _group(panel, "objects")
        assert not row.go_button.isEnabled() and not row.reset_button.isEnabled()
        row.set_button.click()
        window.map_view.on_unit_pick((1, gen.REF_ARCHER_A))
        assert row.go_button.isEnabled() and row.reset_button.isEnabled()
    finally:
        conftest.close_window(window)


def test_a_click_with_no_drag_sets_a_one_tile_area() -> None:
    window = _window()
    try:
        view = window.map_view
        _arm(window, "area")
        _press(view, _tile_pos(view, (12, 7)))
        _release(view, _tile_pos(view, (12, 7)))
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (12, 7, 12, 7)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize(
    ("end", "expected"),
    [((131.5, 125.5), (100, 90, 119, 119)), ((-6.5, 95.5), (0, 90, 100, 95))],
)
def test_a_drag_past_the_map_edge_clamps_to_the_edge(style, end, expected) -> None:
    from PyQt5.QtCore import QEvent

    window = _window(style)
    try:
        view = window.map_view
        _arm(window, "area")
        _press(view, _tile_pos(view, (100, 90)))
        past = _ground_pos(view, *end)
        _move(view, past)
        preview = view._tile_pick_item
        assert preview is not None and not preview.path().isEmpty(), "the band follows past the edge"
        # Leaving the widget mid-drag cancels nothing: Qt's grab still routes the release here.
        view.leaveEvent(QEvent(QEvent.Leave))
        assert view._tile_pick_item is not None
        _release(view, past)
        assert _values(window, "effect", PATROL_EFFECT, AREA) == expected
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_drag_ending_on_a_raised_tile_takes_that_tile_not_the_ground_under_it(style) -> None:
    from PyQt5.QtCore import QPointF
    from PyQt5.QtWidgets import QApplication

    window = _window(style)
    try:
        for tile in window.scenario.map_manager.terrain:
            if 60 <= tile.x < 70 and 60 <= tile.y < 70:
                tile.elevation = 6
        window.refresh_map()
        QApplication.processEvents()
        view = window.map_view
        assert view._terrain_style == MAP_STYLE[style]
        end = _tile_pos(view, (65, 65))
        scene = view.mapToScene(end.toPoint())
        ground = view.ground_map_point(QPointF(scene))
        assert (int(ground[0]), int(ground[1])) != (65, 65), "the raise must move the tile off its ground point"
        _arm(window, "area")
        _press(view, _tile_pos(view, (50, 52)))
        _move(view, end)
        _release(view, end)
        assert _values(window, "effect", PATROL_EFFECT, AREA) == (50, 52, 65, 65)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("group", ["location", "area"])
@pytest.mark.parametrize("how", ["escape", "right_click", "escape_mid_drag"])
def test_esc_and_right_click_cancel_without_writing(group, how) -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        view = window.map_view
        before = _values(window, "effect", PATROL_EFFECT, AREA + LOCATION)
        records = _records(window)
        row = _arm(window, group)
        if how == "escape_mid_drag":
            if group == "area":
                _press(view, _tile_pos(view, (40, 50)))
                _move(view, _tile_pos(view, (20, 30)))
            _escape(view)
            _release(view, _tile_pos(view, (20, 30)))
        elif how == "escape":
            _escape(view)
        else:
            _press(view, _tile_pos(view, (40, 50)), Qt.RightButton)
        assert window._tile_picker is None and view.tile_picker_kind() is None
        assert view._tile_pick_item is None
        assert not row.set_button.isChecked()
        assert _records(window) == records
        assert _values(window, "effect", PATROL_EFFECT, AREA + LOCATION) == before
    finally:
        conftest.close_window(window)


def test_the_rect_preview_shows_with_the_trigger_overlay_turned_off() -> None:
    window = _window()
    try:
        view = window.map_view
        window.trigger_overlay_action.setChecked(False)
        assert view.trigger_overlay_items() and not any(i.isVisible() for i in view.trigger_overlay_items())
        _arm(window, "area")
        _press(view, _tile_pos(view, (40, 50)))
        _move(view, _tile_pos(view, (20, 30)))
        preview = view._tile_pick_item
        assert preview is not None and preview.isVisible()
        assert preview.zValue() == view.TRIGGER_AREA_Z
        assert preview.path().boundingRect() == view._rect_ring_path((20, 30, 41, 51)).boundingRect()
    finally:
        conftest.close_window(window)


def test_set_location_previews_the_hovered_tile_with_the_location_mark() -> None:
    window = _window()
    try:
        view = window.map_view
        _arm(window, "location")
        _move(view, _tile_pos(view, (33, 44)))
        preview = view._tile_pick_item
        assert preview is not None and preview.isVisible()
        assert preview.path().boundingRect() == view._location_mark((33, 44)).boundingRect()
    finally:
        conftest.close_window(window)


def test_a_restyle_mid_drag_drops_the_drag_without_a_dangling_preview() -> None:
    """set_source()'s scene().clear() destroys the preview item; the held
    reference must go with it, or the next move calls into a deleted object."""
    from PyQt5.QtWidgets import QApplication

    window = _window("Stepped")
    try:
        view = window.map_view
        records = _records(window)
        _arm(window, "area")
        _press(view, _tile_pos(view, (40, 50)))
        _move(view, _tile_pos(view, (30, 40)))
        window.terrain_style_combo.setCurrentText("Sloped")
        window.refresh_map()
        QApplication.processEvents()
        assert view.tile_picker_kind() == "rect", "a same-document restyle keeps the pick armed"
        _move(view, _tile_pos(view, (20, 30)))
        _release(view, _tile_pos(view, (20, 30)))
        assert _records(window) == records, "the dropped drag commits nothing"
    finally:
        conftest.close_window(window)


def test_clear_image_forgets_the_preview_item() -> None:
    window = _window()
    try:
        view = window.map_view
        _arm(window, "area")
        _press(view, _tile_pos(view, (40, 50)))
        assert view._tile_pick_item is not None
        view.clear_image()
        assert view._tile_pick_item is None and view._tile_pick_anchor is None
        view.set_tile_picker(None)  # would removeItem() a deleted item
    finally:
        conftest.close_window(window)


# -- arming, exclusion and every disarm site -------------------------------------


def test_arming_a_tile_pick_disarms_the_unit_pick_and_back() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        view = window.map_view
        _arm(window, "area")
        objects = _group(panel, "objects").set_button
        objects.click()
        assert window._unit_picker == (gen.REFERENCE_TRIGGER, "effect", PATROL_EFFECT, "selected_object_ids", True)
        assert window._tile_picker is None and view.tile_picker_kind() is None
        assert view.unit_picker_active() and not _group(panel, "area").set_button.isChecked()
        _group(panel, "location").set_button.click()
        assert window._unit_picker is None and not view.unit_picker_active()
        assert view.tile_picker_kind() == "point" and not objects.isChecked()
    finally:
        conftest.close_window(window)


def test_a_tile_pick_and_the_garrison_pick_disarm_each_other() -> None:
    from test_unit_edit_viewer import _close, _place_tower
    from test_unit_edit_viewer import _window as _units_window

    window = _units_window()
    try:
        tower = _place_tower(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        garrison = window.units_panel.garrison_pick_button
        garrison.click()
        assert window._garrison_picker is not None
        target = (0, "effect", 0, "area", False, "rect")
        window.arm_tile_picker(target)
        assert window._garrison_picker is None and not garrison.isChecked()
        assert window._tile_picker == target and not window.map_view.unit_picker_active()
        garrison.click()
        assert window._garrison_picker is not None and window.map_view.unit_picker_active()
        assert window._tile_picker is None and window.map_view.tile_picker_kind() is None
    finally:
        _close(window)


@pytest.mark.parametrize(
    "how", ["button", "mode", "entry", "trigger", "tool", "undo", "load", "close", "garrison_disarm_path"]
)
def test_every_disarm_site_disarms_a_tile_pick(how) -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        view = window.map_view
        if how == "undo":
            window.set_entry_field_group(gen.REFERENCE_TRIGGER, ("effect", CREATE_EFFECT), {"location_x": 3}, "Set Location")
        records = _records(window)
        before = _values(window, "effect", PATROL_EFFECT, AREA)
        row = _arm(window, "area")
        if how == "button":
            row.set_button.click()
        elif how == "mode":
            window.mode_combo.setCurrentText("View")
        elif how == "entry":
            panel.select_entry("effect", CREATE_EFFECT)
        elif how == "trigger":
            panel.select_trigger(0)
        elif how == "tool":
            window._select_tool("select")
        elif how == "undo":
            window.undo()
        elif how == "load":
            window.load_scenario(FIXTURE_PATH)
        elif how == "close":
            window.close_scenario()
        else:
            window.disarm_unit_picker()
        assert window._tile_picker is None and view.tile_picker_kind() is None
        if how not in ("load", "close", "undo"):
            assert _records(window) == records
            assert _values(window, "effect", PATROL_EFFECT, AREA) == before
    finally:
        conftest.close_window(window)


def test_a_pick_against_an_entry_no_longer_shown_writes_nothing() -> None:
    gen = _gen()
    window = _window()
    try:
        records = _records(window)
        window.arm_tile_picker((gen.REFERENCE_TRIGGER, "effect", PATROL_EFFECT, "location", False, "point"))
        window.trigger_panel.select_entry("effect", CREATE_EFFECT)  # focus change: disarmed already
        window.map_view.on_tile_pick((1, 2))
        window._tile_picker = (gen.REFERENCE_TRIGGER, "effect", PATROL_EFFECT, "location", False, "point")
        window.map_view.on_tile_pick((1, 2))  # armed, but the form shows another entry
        assert _records(window) == records
        assert window._tile_picker is None
    finally:
        conftest.close_window(window)


# -- Go to -----------------------------------------------------------------------


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize(
    ("index", "group", "expected"),
    [
        (PATROL_EFFECT, "location", (60, 50)),
        (PATROL_EFFECT, "area", (32, 43)),  # (30, 40, 34, 46)'s integer centre
        (5, "objects", (51, 56)),  # archers at (50, 55) and (52, 57)
        (4, "area", (59, 59)),  # the whole map
    ],
)
def test_go_to_centres_the_map_on_the_group(style, index, group, expected) -> None:
    window = _window(style)
    try:
        panel = window.trigger_panel
        view = window.map_view
        panel.select_entry("effect", index)
        # Zoomed in: at fit-to-window there may be nothing to scroll.
        view.scale(4.0, 4.0)
        view.center_on_tile(100, 10)
        assert view.viewport_centre_tile() != expected
        records = _records(window)
        _group(panel, group).go_button.click()
        assert view.viewport_centre_tile() == expected
        assert window.status_log.toPlainText().splitlines()[-1].endswith(f"({expected[0]}, {expected[1]})")
        assert _records(window) == records
    finally:
        conftest.close_window(window)
