"""GH #59's Create Objects tool: a Triggers-mode brush that stamps one copy of
the current Create Object effect per tile of the footprint, as one undo record.

Driven through a real ViewerWindow on the trigger fixture, whose triggers carry
no Create Object effect of their own, so each test adds its template through
the same funnels the panel uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

CREATE_OBJECT = 11
TREE_CONST = 349  # Oak tree; any const works, the stamp never reads it
TEMPLATE_TILE = (5, 5)
# Off-centre by the east edge (map is 120 wide): the circle is clipped on one
# side only, so an x/y swap in the funnel lands on a different tile set.
EDGE_CLICK = (118, 60)


def _window():
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    window = conftest.shown_window(1500, 900)
    for key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
        QTest.keyRelease(window, key)
    window.load_scenario(TRIGGER_FIXTURE)
    window.mode_combo.setCurrentText("Triggers")
    panel = window.trigger_panel
    panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
    assert panel.current_trigger_index() == 0
    return window


def _add_template(window, trigger_index: int = 0) -> int:
    """A Create Object effect on `trigger_index`, selected as the current entry.
    Returns its effect index."""
    from descape.trigger_fields import INT, FieldSpec

    window.entry_structural_edit("new", trigger_index, "effect", -1, CREATE_OBJECT)
    effect_index = len(window.trigger_edits.manager().triggers[trigger_index].effects) - 1
    for name, value in (
        ("object_list_unit_id", TREE_CONST),
        ("source_player", 0),
        ("location_x", TEMPLATE_TILE[0]),
        ("location_y", TEMPLATE_TILE[1]),
        ("facet", 3),
    ):
        window.set_entry_field(trigger_index, "effect", effect_index, FieldSpec(name, INT), value)
    window.trigger_panel.select_entry("effect", effect_index)
    assert window.trigger_panel.current_entry_ref() == ("effect", effect_index)
    return effect_index


def _set_brush(window, size: int, shape: str) -> None:
    window.brush_size_spin.setValue(size)
    window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(shape))


def _effects(window, trigger_index: int = 0):
    return window.trigger_edits.manager().triggers[trigger_index].effects


def _fields(effect) -> dict:
    """Create Object's vocabulary attributes (1.58)."""
    return {
        name: getattr(effect, name)
        for name in (
            "effect_type", "object_list_unit_id", "source_player", "location_x",
            "location_y", "item_id", "facet", "disable_sound",
        )
    }


# -- registry and toolbar -----------------------------------------------------


def test_the_tool_shows_only_in_triggers_mode_with_its_brush_widgets() -> None:
    window = _window()
    try:
        assert window.create_objects_action.isVisible()
        assert window.create_objects_action.isEnabled()
        window.create_objects_action.setChecked(True)
        assert window._current_tool == "create_objects"
        assert window.brush_size_spin_action.isVisible()
        assert window.brush_shape_combo_action.isVisible()
        assert window.tool_param_separator_action.isVisible()
        # No terrain/level param rides along with it.
        assert not window.paint_trees_param_action.isVisible()
        assert not window.level_param_spin_action.isVisible()

        window.mode_combo.setCurrentText("Terrain")
        assert not window.create_objects_action.isVisible()
        assert window.pan_action.isChecked(), "leaving Triggers must not leave the tool live"
        assert not window.brush_size_spin_action.isVisible()
    finally:
        conftest.close_window(window)


def test_the_brush_reaches_the_hover_preview() -> None:
    from descape import brush

    window = _window()
    try:
        _set_brush(window, 5, brush.BRUSH_SHAPE_CIRCLE)
        window.create_objects_action.setChecked(True)
        window.map_view.refresh_highlight(EDGE_CLICK)
        assert window.map_view._highlight_key == (*EDGE_CLICK, 5, brush.BRUSH_SHAPE_CIRCLE)
    finally:
        conftest.close_window(window)


# -- the stamp ----------------------------------------------------------------


def test_a_click_emits_one_effect_per_footprint_tile() -> None:
    from descape import brush

    window = _window()
    try:
        template_index = _add_template(window)
        template_before = _fields(_effects(window)[template_index])
        records_before = len(window.edit_history.records)
        _set_brush(window, 5, brush.BRUSH_SHAPE_CIRCLE)

        window.stamp_create_objects(*EDGE_CLICK)

        footprint = brush.brush_tiles(*EDGE_CLICK, 5, brush.BRUSH_SHAPE_CIRCLE, 120, 120)
        assert len(footprint) < len(brush.brush_offsets(5, brush.BRUSH_SHAPE_CIRCLE)), "not clipped"
        effects = _effects(window)
        emitted = effects[template_index + 1:]
        assert [(e.location_x, e.location_y) for e in emitted] == footprint
        for effect in emitted:
            fields = _fields(effect)
            del fields["location_x"], fields["location_y"]
            expected = dict(template_before)
            del expected["location_x"], expected["location_y"]
            assert fields == expected
        assert _fields(effects[template_index]) == template_before, "the template must be untouched"
        assert len(window.edit_history.records) == records_before + 1, "one click, one record"
        assert window.trigger_panel.current_entry_ref() == ("effect", len(effects) - 1)
        assert window.trigger_edits.is_dirty(0)
    finally:
        conftest.close_window(window)


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def test_a_second_click_on_the_same_spot_stacks_nothing() -> None:
    from descape import brush

    window = _window()
    try:
        _add_template(window)
        _set_brush(window, 3, brush.BRUSH_SHAPE_SQUARE)
        window.stamp_create_objects(50, 50)
        count = len(_effects(window))
        records = len(window.edit_history.records)
        # GH #59: each click reports its count and how many tiles it skipped.
        assert _last_status(window) == "Created 9 Create Object effects"

        window.stamp_create_objects(50, 50)
        assert len(_effects(window)) == count
        assert len(window.edit_history.records) == records, "an all-duplicate click records nothing"
        assert _last_status(window) == "Create Objects: all 9 tiles already have this object"

        # One column over: only the new column's three tiles are emitted.
        window.stamp_create_objects(51, 50)
        assert len(_effects(window)) == count + 3
        assert {(e.location_x, e.location_y) for e in _effects(window)[-3:]} == {
            (52, 49), (52, 50), (52, 51)
        }
        assert _last_status(window) == "Created 3 Create Object effects (6 tiles already had one)"
    finally:
        conftest.close_window(window)


def test_the_templates_own_tile_is_a_duplicate() -> None:
    from descape import brush

    window = _window()
    try:
        _add_template(window)
        _set_brush(window, 1, brush.BRUSH_SHAPE_SQUARE)
        count = len(_effects(window))
        records = len(window.edit_history.records)
        window.stamp_create_objects(*TEMPLATE_TILE)
        assert len(_effects(window)) == count
        assert len(window.edit_history.records) == records
    finally:
        conftest.close_window(window)


def test_a_different_object_on_a_tile_is_not_a_duplicate() -> None:
    from descape import brush
    from descape.trigger_fields import INT, FieldSpec

    window = _window()
    try:
        index = _add_template(window)
        _set_brush(window, 1, brush.BRUSH_SHAPE_SQUARE)
        window.stamp_create_objects(20, 20)
        count = len(_effects(window))
        # Re-point the template at another object: the tree already stamped at
        # (20, 20) is not the same object, so that tile is no duplicate.
        window.set_entry_field(0, "effect", index, FieldSpec("object_list_unit_id", INT), TREE_CONST + 1)
        window.trigger_panel.select_entry("effect", index)
        window.stamp_create_objects(20, 20)
        assert len(_effects(window)) == count + 1
        assert _effects(window)[-1].object_list_unit_id == TREE_CONST + 1
    finally:
        conftest.close_window(window)


# -- refusals -----------------------------------------------------------------


def _assert_refused(window, x: int = 30, y: int = 30) -> None:
    count = len(_effects(window))
    records = len(window.edit_history.records)
    window.stamp_create_objects(x, y)
    assert len(_effects(window)) == count
    assert len(window.edit_history.records) == records
    assert window.status_log.toPlainText().splitlines()[-1].startswith("Create Objects:")


def test_a_non_create_object_effect_is_refused(tmp_path: Path) -> None:
    from descape.scenario_write import write_scenario

    window = _window()
    try:
        window.trigger_panel.select_entry("effect", 0)  # trigger 0's effect is type 20
        assert window.trigger_panel.current_entry_ref() == ("effect", 0)
        window.stamp_create_objects(30, 30)
        # The type check needs the model, so it is built, but nothing is
        # recorded and the document still saves byte-identically.
        assert window.edit_history.records == []
        assert len(_effects(window)) == 1
        out = tmp_path / "refused.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        conftest.close_window(window)


def test_a_condition_or_the_trigger_row_is_refused_before_any_model_exists() -> None:
    window = _window()
    try:
        panel = window.trigger_panel
        panel.entry_tree.setCurrentItem(panel.entry_tree.topLevelItem(0))  # the Trigger row
        assert panel.current_entry_ref() is None
        window.stamp_create_objects(30, 30)
        assert window.trigger_edits is None, "a refused click must not build the model"
        assert window.edit_history.records == []
    finally:
        conftest.close_window(window)


def test_an_open_picker_is_refused() -> None:
    window = _window()
    try:
        _add_template(window)
        window.trigger_panel._open_picker(0)
        assert window.trigger_panel.picker_showing()
        _assert_refused(window)
    finally:
        conftest.close_window(window)


def test_several_selected_triggers_are_refused() -> None:
    window = _window()
    try:
        _add_template(window)
        window.trigger_panel.tree.topLevelItem(1).setSelected(True)
        assert len(window.trigger_panel.selected_trigger_indices()) == 2
        _assert_refused(window)
    finally:
        conftest.close_window(window)


# -- routing, undo, write path ------------------------------------------------


def test_a_real_click_reaches_the_stamp() -> None:
    """Trap 4: a click_only tool must reach on_click_edit, not fall through to
    Pan's drag or a mode branch above it."""
    from PyQt5.QtCore import QEvent, Qt

    from descape import brush

    window = _window()
    try:
        _add_template(window)
        _set_brush(window, 1, brush.BRUSH_SHAPE_SQUARE)
        window.create_objects_action.setChecked(True)
        map_view = window.map_view
        count = len(_effects(window))
        pos = conftest.polygon_viewport_pos(map_view, 40, 41)
        map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
        map_view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))
        assert len(_effects(window)) == count + 1
        assert (_effects(window)[-1].location_x, _effects(window)[-1].location_y) == (40, 41)
    finally:
        conftest.close_window(window)


def test_one_undo_removes_the_whole_stamp_and_redo_restores_it() -> None:
    from descape import brush

    window = _window()
    try:
        _add_template(window)
        _set_brush(window, 3, brush.BRUSH_SHAPE_SQUARE)
        before = [_fields(e) for e in _effects(window)]
        window.stamp_create_objects(60, 60)
        after = [_fields(e) for e in _effects(window)]
        assert len(after) == len(before) + 9

        window.undo()
        assert [_fields(e) for e in _effects(window)] == before
        window.redo()
        assert [_fields(e) for e in _effects(window)] == after
    finally:
        conftest.close_window(window)


def test_a_stamped_trigger_round_trips_through_save(tmp_path: Path) -> None:
    """The template is saved and reopened first, so the stamp is the only edit
    that dirties trigger 0: without content_touched its blob splices back."""
    from descape import brush, scenario_io
    from descape.scenario_write import write_scenario

    seeded = tmp_path / "seeded.aoe2scenario"
    out = tmp_path / "stamped.aoe2scenario"
    window = _window()
    try:
        template_index = _add_template(window)
        write_scenario(window.scenario, seeded, triggers=window.trigger_edits)
        window.edit_history.mark_saved()
        window.load_scenario(seeded)
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        panel.select_entry("effect", template_index)
        assert window.trigger_edits is None

        _set_brush(window, 5, brush.BRUSH_SHAPE_CIRCLE)
        window.stamp_create_objects(*EDGE_CLICK)
        expected = [_fields(e) for e in _effects(window)]
        stamped = expected[template_index + 1:]
        assert len(stamped) == len(brush.brush_tiles(*EDGE_CLICK, 5, brush.BRUSH_SHAPE_CIRCLE, 120, 120))
        assert len({(f["location_x"], f["location_y"]) for f in stamped}) == len(stamped)

        write_scenario(window.scenario, out, triggers=window.trigger_edits)
    finally:
        conftest.close_window(window)

    manager = scenario_io.parse_triggers(scenario_io.load_map_and_units(out))
    assert manager is not None
    assert [_fields(e) for e in manager.triggers[0].effects] == expected
