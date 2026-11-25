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
TREE_CONST = 349  # Oak tree, 1x1, so the stamp steps by one tile (TASK-086)
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
        assert window.map_view._highlight_key == (*EDGE_CLICK, 5, brush.BRUSH_SHAPE_CIRCLE, 1)
    finally:
        conftest.close_window(window)


def test_a_square_outline_stamp_writes_its_ring_only() -> None:
    """TASK-257: a non-terrain brush tool reads the shape and thickness through _brush_footprint()."""
    from descape import brush

    window = _window()
    try:
        template_index = _add_template(window)
        _set_brush(window, 5, brush.BRUSH_SHAPE_SQUARE_OUTLINE)
        window.stamp_create_objects(50, 50)
        emitted = _effects(window)[template_index + 1:]
        assert len(emitted) == 16
        assert (50, 50) not in {(e.location_x, e.location_y) for e in emitted}
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


# -- TASK-086: a multi-tile object steps by its footprint ---------------------

HOUSE_CONST = 70  # 2x2
GATE_X_CONST = 64  # Gate (Up.), the stone gate: 4 tiles along x, 1 along y


def _retarget(window, index: int, const: int, location: tuple[int, int] | None = None) -> None:
    from descape.trigger_fields import INT, FieldSpec

    window.set_entry_field(0, "effect", index, FieldSpec("object_list_unit_id", INT), const)
    if location is not None:
        for name, value in zip(("location_x", "location_y"), location, strict=True):
            window.set_entry_field(0, "effect", index, FieldSpec(name, INT), value)
    window.trigger_panel.select_entry("effect", index)


def _stamped(window, after: int) -> list[tuple[int, int]]:
    return [(e.location_x, e.location_y) for e in _effects(window)[after + 1:]]


def test_a_house_stamps_one_copy_per_2x2_footprint() -> None:
    from descape import brush

    window = _window()
    try:
        index = _add_template(window)
        _retarget(window, index, HOUSE_CONST)
        _set_brush(window, 4, brush.BRUSH_SHAPE_SQUARE)
        records = len(window.edit_history.records)
        window.stamp_create_objects(50, 50)
        assert _stamped(window, index) == [(50, 50), (52, 50), (50, 52), (52, 52)]
        assert len(window.edit_history.records) == records + 1
        assert _last_status(window) == "Created 4 Create Object effects, one per 2x2 footprint"
    finally:
        conftest.close_window(window)


def test_the_stamped_houses_tile_a_block_by_the_games_anchor_rule() -> None:
    """T2g: by GH #369's created_unit_point(), the four footprints abut with no gap or overlap."""
    from descape import brush
    from descape.render_units import NON_BUILDING_SPAN, created_unit_point
    from descape.terrain_palette import tile_span

    window = _window()
    try:
        index = _add_template(window)
        _retarget(window, index, HOUSE_CONST)
        _set_brush(window, 4, brush.BRUSH_SHAPE_SQUARE)
        window.stamp_create_objects(50, 50)
        sx, sy = tile_span(HOUSE_CONST, NON_BUILDING_SPAN)
        covered: list[tuple[int, int]] = []
        for x, y in _stamped(window, index):
            px, py = created_unit_point(HOUSE_CONST, x, y)
            x0, y0 = int(px - sx / 2), int(py - sy / 2)
            own = [(tx, ty) for ty in range(y0, y0 + sy) for tx in range(x0, x0 + sx)]
            assert len(own) == 4
            covered += own
        assert len(covered) == len(set(covered)), "two footprints overlap"
        assert set(covered) == {(tx, ty) for ty in range(49, 53) for tx in range(49, 53)}
    finally:
        conftest.close_window(window)


def test_a_4x1_gate_steps_along_x_only() -> None:
    """T2b, Trap 3: span_x pairs with location_x."""
    from descape import brush

    window = _window()
    try:
        index = _add_template(window)
        _retarget(window, index, GATE_X_CONST)
        _set_brush(window, 9, brush.BRUSH_SHAPE_LINE_X)
        window.stamp_create_objects(50, 50)
        assert _stamped(window, index) == [(46, 50), (50, 50), (54, 50)]
        count = len(_effects(window))
        _set_brush(window, 9, brush.BRUSH_SHAPE_LINE_Y)
        window.stamp_create_objects(50, 60)
        assert _stamped(window, count - 1) == [(50, y) for y in range(56, 65)]
    finally:
        conftest.close_window(window)


def test_a_second_house_click_skips_the_copies_that_would_overlap() -> None:
    """T2c: the duplicate skip is a footprint-overlap skip for a multi-tile object."""
    from descape import brush

    window = _window()
    try:
        index = _add_template(window)
        _retarget(window, index, HOUSE_CONST)
        _set_brush(window, 4, brush.BRUSH_SHAPE_SQUARE)
        window.stamp_create_objects(50, 50)
        count = len(_effects(window))
        window.stamp_create_objects(53, 50)
        assert _stamped(window, count - 1) == [(55, 50), (55, 52)]
        assert _last_status(window) == (
            "Created 2 Create Object effects, one per 2x2 footprint (2 would overlap one already there)"
        )
        count = len(_effects(window))
        records = len(window.edit_history.records)
        window.stamp_create_objects(50, 50)
        assert len(_effects(window)) == count
        assert len(window.edit_history.records) == records
        assert _last_status(window) == (
            "Create Objects: every 2x2 spot here would overlap this object already there"
        )
    finally:
        conftest.close_window(window)


def test_an_unset_template_location_blocks_no_tile() -> None:
    """T2d, Trap 1: a (-1, -1) template must not overlap a House at (0, 0)."""
    from descape import brush

    window = _window()
    try:
        index = _add_template(window)
        _retarget(window, index, HOUSE_CONST, location=(-1, -1))
        _set_brush(window, 1, brush.BRUSH_SHAPE_SQUARE)
        window.stamp_create_objects(0, 0)
        assert _stamped(window, index) == [(0, 0)]
    finally:
        conftest.close_window(window)


def test_the_hover_highlight_shows_only_the_anchor_tiles() -> None:
    """T2e: the preview and the stamp share lattice_tiles(), so they cannot disagree."""
    from descape import brush

    window = _window()
    try:
        index = _add_template(window)
        _retarget(window, index, HOUSE_CONST)
        _set_brush(window, 4, brush.BRUSH_SHAPE_SQUARE)
        window.create_objects_action.setChecked(True)
        map_view = window.map_view
        calls = []
        original = map_view._set_highlight_tiles

        def recording(tiles, key):
            calls.append((list(tiles), key))
            original(tiles, key)

        map_view._set_highlight_tiles = recording
        map_view.refresh_highlight((50, 50))
        assert map_view._highlight_key == ("stamp", 50, 50, 4, brush.BRUSH_SHAPE_SQUARE, 1, (2, 2))
        assert calls[-1][0] == [(50, 50), (52, 50), (50, 52), (52, 52)]
        records = len(window.edit_history.records)
        map_view.refresh_highlight((51, 50))
        assert len(window.edit_history.records) == records, "a hover records nothing"
        # Back to a 1x1 template: the plain key again.
        _retarget(window, index, TREE_CONST)
        map_view.refresh_highlight((50, 50))
        assert map_view._highlight_key == (50, 50, 4, brush.BRUSH_SHAPE_SQUARE, 1)
        assert len(calls[-1][0]) == 16
    finally:
        conftest.close_window(window)


def test_the_hover_step_query_builds_no_edit_model(tmp_path: Path) -> None:
    """Trap 2: the hover never calls _ensure_trigger_edits(). A saved House template is
    reopened, so the file has one and no model exists yet."""
    from descape import brush
    from descape.scenario_write import write_scenario

    seeded = tmp_path / "house.aoe2scenario"
    window = _window()
    try:
        index = _add_template(window)
        _retarget(window, index, HOUSE_CONST)
        write_scenario(window.scenario, seeded, triggers=window.trigger_edits)
        window.edit_history.mark_saved()
        window.load_scenario(seeded)
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        panel.select_entry("effect", index)
        assert window.trigger_edits is None
        _set_brush(window, 4, brush.BRUSH_SHAPE_SQUARE)
        window.create_objects_action.setChecked(True)
        assert window._create_objects_step() == (2, 2)
        window.map_view.refresh_highlight((50, 50))
        assert window.map_view._highlight_key[0] == "stamp"
        assert window.trigger_edits is None, "a hover must not build the edit model"
        panel.select_entry("effect", 0)  # the fixture's own type-20 effect
        assert window._create_objects_step() == (1, 1)
    finally:
        conftest.close_window(window)


def test_the_toolbar_and_hint_tooltips_name_the_footprint_step() -> None:
    from PyQt5.QtWidgets import QLabel

    window = _window()
    try:
        wording = "one per footprint for a bigger object"
        assert wording in window.create_objects_action.toolTip()
        _add_template(window)
        panel = window.trigger_panel
        hints = [w for w in window.findChildren(QLabel) if w.text() == panel._CREATE_OBJECTS_HINT]
        assert hints, "the Create Object form shows its Create Objects hint"
        assert all(wording in hint.toolTip() for hint in hints)
    finally:
        conftest.close_window(window)
