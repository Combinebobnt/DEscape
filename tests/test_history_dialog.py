"""GH #30's History window: the dialog itself, and the ViewerWindow wiring
that feeds it. EditHistory's own jump_to/span_kinds/on_change rules are
covered without Qt in tests/test_edit_history.py.

Same offscreen technique tests/test_clipboard_dialog.py documents; every
ViewerWindow() here must call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import pytest

from descape.history_dialog import OPENED_FILE_ID, EditHistoryDialog

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN_A, _TERRAIN_B = 2, 15  # BEACH, GRASS_1 -- present in every DE version


def _rows(count: int):
    """Row ids deliberately unequal to their own row numbers, so an
    index-for-id confusion cannot pass."""
    return [(1000 + i, f"edit {i}", "tile") for i in range(count)]


def _dialog(**callbacks):
    conftest.ensure_qapp()
    return EditHistoryDialog(None, **callbacks)


def _texts(dialog, column: int = 1):
    return [
        dialog.tree.topLevelItem(row).text(column) for row in range(dialog.tree.topLevelItemCount())
    ]


# -- the dialog on its own ---------------------------------------------------


def test_set_rows_prepends_the_opened_file_row() -> None:
    from PyQt5.QtCore import Qt

    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 3, 0)
        assert dialog.tree.topLevelItemCount() == 4, "three records plus 'Opened file'"
        assert _texts(dialog) == ["Opened file", "edit 0", "edit 1", "edit 2"]
        ids = [dialog.tree.topLevelItem(r).data(0, Qt.UserRole) for r in range(4)]
        assert ids == [OPENED_FILE_ID, 1000, 1001, 1002]
        targets = [dialog.tree.topLevelItem(r).data(1, Qt.UserRole) for r in range(4)]
        assert targets == [0, 1, 2, 3], "row N stands for cursor N"
    finally:
        dialog.close()


def test_empty_state_label_shows_with_no_records() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows([], 0, 0)
        assert dialog.tree.topLevelItemCount() == 1, "'Opened file' alone"
        assert dialog.status.text() == dialog._EMPTY
    finally:
        dialog.close()


def test_the_current_row_is_bold_and_undone_rows_are_greyed() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 1, 0)
        bold = [dialog.tree.topLevelItem(r).font(1).bold() for r in range(4)]
        assert bold == [False, True, False, False]
        normal = dialog.tree.topLevelItem(0).foreground(1).color()
        greyed = [dialog.tree.topLevelItem(r).foreground(1).color() for r in (2, 3)]
        assert all(c != normal for c in greyed), "rows above the cursor read as undone"
        assert dialog.tree.topLevelItem(1).foreground(1).color() == normal
    finally:
        dialog.close()


def test_the_opened_file_row_is_bold_at_cursor_zero() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(2), 0, 0)
        assert dialog.tree.topLevelItem(0).font(1).bold()
        assert not dialog.tree.topLevelItem(1).font(1).bold()
    finally:
        dialog.close()


def test_the_saved_marker_sits_on_the_saved_row_only() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 3, 2)
        markers = [bool(t) for t in _texts(dialog, 0)]
        assert markers == [False, False, True, False]
    finally:
        dialog.close()


def test_no_saved_marker_when_the_saved_state_fell_off_the_cap() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 3, None)
        assert not any(_texts(dialog, 0))
    finally:
        dialog.close()


def test_selection_survives_a_repopulate_by_id_not_by_row() -> None:
    """The discriminating case: the trim drops the oldest record, so every
    surviving row shifts down one while its id stays put."""
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(3), 3, 0)
        dialog.select_row(1002)
        assert dialog.selected_row_id() == 1002
        dialog.set_rows(_rows(3)[1:], 2, None)
        assert dialog.selected_row_id() == 1002
        assert dialog.selected_target() == 2, "one row earlier after the trim"
    finally:
        dialog.close()


def test_double_click_and_the_button_both_report_the_selected_target() -> None:
    seen = []
    dialog = _dialog(on_jump=seen.append)
    try:
        dialog.set_rows(_rows(3), 3, 0)
        dialog.select_row(1000)
        dialog.jump_button.click()
        assert seen == [1], "row 'edit 0' is cursor 1"
        dialog.select_row(OPENED_FILE_ID)
        dialog.tree.itemDoubleClicked.emit(dialog.tree.topLevelItem(0), 0)
        assert seen == [1, 0]
    finally:
        dialog.close()


def test_selecting_a_row_does_not_jump_on_its_own() -> None:
    """Only a double-click or the button may move the history: a plain
    keyboard walk down the list would otherwise undo the document."""
    seen = []
    dialog = _dialog(on_jump=seen.append)
    try:
        dialog.set_rows(_rows(3), 3, 0)
        dialog.select_row(1000)
        dialog.set_rows(_rows(3), 1, 0)
        assert seen == []
    finally:
        dialog.close()


def test_the_jump_button_gates_on_a_selection() -> None:
    dialog = _dialog()
    try:
        dialog.set_rows(_rows(2), 2, 0)
        dialog.tree.setCurrentItem(None)
        assert not dialog.jump_button.isEnabled()
        dialog.select_row(1001)
        assert dialog.jump_button.isEnabled()
    finally:
        dialog.close()


# -- the window wiring -------------------------------------------------------


def _window():
    window = conftest.terrain_edit_window()
    window._on_tool_selected("draw")
    return window


def _paint(window, x: int, y: int, terrain_id: int) -> None:
    window.terrain_panel.set_terrain(terrain_id)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()


def test_the_dialog_is_non_modal_and_lists_every_record() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        _paint(window, 4, 4, _TERRAIN_B)
        window._show_history_dialog()
        dialog = window._history_dialog
        assert dialog is not None
        assert not dialog.isModal()
        assert dialog.tree.topLevelItemCount() == 3, "two records plus 'Opened file'"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_new_edit_with_the_dialog_open_refreshes_it_through_on_change() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._show_history_dialog()
        dialog = window._history_dialog
        assert dialog.tree.topLevelItemCount() == 2
        _paint(window, 4, 4, _TERRAIN_B)
        assert dialog.tree.topLevelItemCount() == 3, "no push site had to call the refresh"
        assert dialog.tree.topLevelItem(2).font(1).bold(), "the newest row is current"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_jumping_to_zero_restores_the_tiles_and_the_clean_title() -> None:
    window = _window()
    try:
        mm = window.scenario.map_manager
        before = mm.get_tile(3, 3).terrain_id
        window.edit_history.mark_saved()
        _paint(window, 3, 3, _TERRAIN_A)
        _paint(window, 4, 4, _TERRAIN_B)
        assert window.windowTitle().startswith("*")

        window._show_history_dialog()
        window._on_history_jump(0)
        assert window.edit_history.cursor == 0
        assert mm.get_tile(3, 3).terrain_id == before
        assert not window.windowTitle().startswith("*"), "back onto the saved cursor"
        assert not window.undo_action.isEnabled()
        assert window.redo_action.isEnabled()

        window._on_history_jump(2)
        assert mm.get_tile(3, 3).terrain_id == _TERRAIN_A
        assert mm.get_tile(4, 4).terrain_id == _TERRAIN_B
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_jump_reports_the_distance_and_the_entry_it_landed_on() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        _paint(window, 4, 4, _TERRAIN_B)
        window._on_history_jump(0)
        assert "Jumped back 2 steps to: Opened file" in window.status_log.toPlainText()
        window._on_history_jump(1)
        assert "Jumped forward 1 step to:" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_jump_to_the_current_cursor_reports_rather_than_moving() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._on_history_jump(1)
        assert window.edit_history.cursor == 1
        assert "Already at that history entry" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_stale_out_of_range_row_is_refused_not_raised() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._on_history_jump(9)
        assert window.edit_history.cursor == 1
        assert "Jump refused" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def _open_dialog(window) -> EditHistoryDialog:
    """Through the menu action, found by type rather than the private handle."""
    window.history_action.trigger()
    dialog = window.findChild(EditHistoryDialog)
    assert dialog is not None and dialog.isVisible()
    return dialog


def _click_jump(window, target: int) -> None:
    """Row N stands for cursor N, so select row `target` and press Jump Here."""
    dialog = _open_dialog(window)
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(target))
    assert dialog.selected_target() == target
    dialog.jump_button.click()


def test_a_jump_mid_stroke_is_refused_and_leaves_the_cursor() -> None:
    """GH #30 step 14: the dialog is non-modal, so Jump Here can be pressed
    while a paint stroke is still open."""
    window = _window()
    stroking = False
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        _paint(window, 4, 4, _TERRAIN_B)
        window.terrain_panel.set_terrain(_TERRAIN_A)
        window.on_edit_stroke_start()
        stroking = True
        window.on_edit_stroke_tile(5, 5, 0)
        assert window.edit_history.in_stroke

        _click_jump(window, 0)

        assert "Jump: an edit is still in progress" in window.status_log.toPlainText()
        assert window.edit_history.cursor == 2
        assert window.scenario.map_manager.get_tile(3, 3).terrain_id == _TERRAIN_A
    finally:
        if stroking:
            window.on_edit_stroke_end()
        window.edit_history.mark_saved()
        window.close()


def test_the_dialog_tracks_a_jump_made_from_the_menu_undo() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        _paint(window, 4, 4, _TERRAIN_B)
        window._show_history_dialog()
        dialog = window._history_dialog
        window.undo()
        bold = [dialog.tree.topLevelItem(r).font(1).bold() for r in range(3)]
        assert bold == [False, True, False], "undo moved the bold row without a manual refresh"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_closing_the_file_empties_the_dialog() -> None:
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._show_history_dialog()
        dialog = window._history_dialog
        window.edit_history.mark_saved()
        window.close_scenario()
        assert dialog.tree.topLevelItemCount() == 1, "reset() reached the dialog"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_history_action_is_registered_and_unbound_by_default() -> None:
    from descape import settings

    ids = [row[0] for row in settings.REBINDABLE_ACTIONS]
    assert "edit_history" in ids
    assert settings.get_keybind("edit_history") == ""


def test_a_hidden_dialog_is_not_rebuilt_but_reopens_current() -> None:
    """Close only hides a QDialog, so the on_change hook would otherwise keep
    rebuilding a few hundred invisible rows on every stroke."""
    window = _window()
    try:
        _paint(window, 3, 3, _TERRAIN_A)
        window._show_history_dialog()
        dialog = window._history_dialog
        dialog.close()
        _paint(window, 4, 4, _TERRAIN_B)
        assert dialog.tree.topLevelItemCount() == 2, "no rebuild while hidden"
        window._show_history_dialog()
        assert dialog.tree.topLevelItemCount() == 3, "reopening catches it up"
        assert dialog.tree.topLevelItem(2).font(1).bold()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #30 steps 8-12: jumps over non-tile records ----------------------------


def _unit_by_ref(window, reference_id: int):
    """(player_id, unit) straight from the scenario, or None once it is gone."""
    for player_id, units in enumerate(window.scenario.unit_manager.units):
        for unit in units:
            if unit.reference_id == reference_id:
                return player_id, unit
    return None


def _assert_inspector_shows(window, reference_id: int) -> None:
    """Every inspector field that has a model counterpart reads the model."""
    from descape import unit_rotation

    found = _unit_by_ref(window, reference_id)
    assert found is not None, f"unit {reference_id} is not in the model"
    player_id, unit = found
    panel = window.units_panel
    assert panel.unit_inspector_grid.isVisibleTo(window.left_stack)
    assert not panel.unit_inspector_empty.isVisibleTo(window.left_stack)
    assert panel.unit_field_labels["reference_id"].text() == str(reference_id)
    assert panel.unit_field_editors["player"].currentData() == player_id
    for field in ("x", "y"):
        editor = panel.unit_field_editors[field]
        assert editor.value() == pytest.approx(getattr(unit, field), abs=10 ** -editor.decimals())
    if unit_rotation.rotation_is_angle(unit.unit_const):
        facing = unit_rotation.rotation_to_facing(unit.rotation, unit_rotation.angle_count_for(unit.unit_const))
        assert panel.unit_field_editors["rotation"].value() == facing
    else:
        assert panel.unit_field_labels["rotation"].text() == f"{unit.rotation:g}"


def _assert_inspector_empty(window) -> None:
    panel = window.units_panel
    assert panel.unit_inspector_empty.isVisibleTo(window.left_stack)
    assert not panel.unit_inspector_grid.isVisibleTo(window.left_stack)


_UNIT_OPERATIONS = ["nudge", "rotate", "variant"]


def _unit_operation(name: str):
    """(reference_id, operate) for one fixture unit edit; imported late, as
    test_unit_edit_viewer needs PyQt5 at import time."""
    from PyQt5.QtCore import Qt
    from test_unit_edit_viewer import _REF_ARCHER_P1, _REF_TREE_OAK

    return {
        "nudge": (_REF_ARCHER_P1, lambda w: w.on_unit_nudge(1, 0, Qt.NoModifier)),
        "rotate": (_REF_ARCHER_P1, lambda w: w.on_unit_rotate(1)),
        "variant": (_REF_TREE_OAK, lambda w: w.on_unit_variant(1)),
    }[name]


@pytest.mark.parametrize("name", _UNIT_OPERATIONS)
def test_a_jump_over_unit_edits_leaves_the_inspector_on_the_model(name: str) -> None:
    from test_unit_edit_viewer import _close, _select
    from test_unit_edit_viewer import _window as _units_window

    reference_id, operate = _unit_operation(name)
    window = _units_window()
    try:
        _select(window, reference_id)
        start = _unit_by_ref(window, reference_id)[1]
        start_fields = (start.x, start.y, start.rotation)
        operate(window)
        operate(window)
        end = window.edit_history.cursor
        assert end == 2, f"{name} did not record two steps"
        _, unit = _unit_by_ref(window, reference_id)
        end_fields = (unit.x, unit.y, unit.rotation)
        assert end_fields != start_fields

        _click_jump(window, 0)
        assert window.edit_history.cursor == 0
        _, unit = _unit_by_ref(window, reference_id)
        assert (unit.x, unit.y, unit.rotation) == start_fields
        _assert_inspector_shows(window, reference_id)

        _click_jump(window, end)
        assert window.edit_history.cursor == end
        _, unit = _unit_by_ref(window, reference_id)
        assert (unit.x, unit.y, unit.rotation) == end_fields
        _assert_inspector_shows(window, reference_id)
    finally:
        _close(window)


def test_a_jump_that_deletes_the_selected_unit_empties_the_inspector() -> None:
    """Delete, jump back so the unit returns, select it, then jump forward
    past its delete: the inspector must not keep showing a unit that is gone."""
    from PyQt5.QtCore import Qt
    from test_unit_edit_viewer import _REF_ARCHER_P1, _REF_VILLAGER_P2, _close, _select
    from test_unit_edit_viewer import _window as _units_window

    window = _units_window()
    try:
        _select(window, _REF_ARCHER_P1)
        window.on_unit_delete(Qt.NoModifier)
        _select(window, _REF_VILLAGER_P2)
        window.on_unit_delete(Qt.NoModifier)
        assert window.edit_history.cursor == 2
        assert _unit_by_ref(window, _REF_ARCHER_P1) is None

        _click_jump(window, 0)
        assert _unit_by_ref(window, _REF_ARCHER_P1) is not None
        assert _unit_by_ref(window, _REF_VILLAGER_P2) is not None
        _select(window, _REF_ARCHER_P1)
        _assert_inspector_shows(window, _REF_ARCHER_P1)

        _click_jump(window, 2)
        assert _unit_by_ref(window, _REF_ARCHER_P1) is None
        _assert_inspector_empty(window)
        assert "No unit selected" in window.units_panel.unit_inspector_empty.text()
    finally:
        _close(window)


def _triggers_window(path):
    window = conftest.shown_window()
    window.load_scenario(path)
    assert window.scenario is not None, "fixture failed to load"
    window.mode_combo.setCurrentText("Triggers")
    return window


def _name_spec():
    from descape import trigger_fields

    return next(spec for spec in trigger_fields.TRIGGER_FIELDS if spec.name == "name")


def _rename_twice(window) -> None:
    window.trigger_panel.select_trigger(1)
    window.set_trigger_field(1, _name_spec(), "Renamed once")
    window.set_trigger_field(1, _name_spec(), "Renamed twice")


def _delete_the_current_and_the_first(window) -> None:
    window.trigger_panel.select_trigger(3)
    window.trigger_structural_edit("delete", [3])
    window.trigger_structural_edit("delete", [0])


_TRIGGER_OPERATIONS = [("rename", _rename_twice), ("structural", _delete_the_current_and_the_first)]


def _assert_tree_matches_the_model(window) -> None:
    """One row per model trigger, each named from its own index, and a
    current trigger that still exists."""
    from PyQt5.QtCore import Qt

    triggers = window.trigger_edits.manager().triggers
    tree = window.trigger_panel.tree
    header = tree.headerItem()
    name_col = next(c for c in range(tree.columnCount()) if header.text(c) == "Trigger")
    assert tree.topLevelItemCount() == len(triggers)
    shown = {}
    for row in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(row)
        shown[item.data(0, Qt.UserRole)] = item.text(name_col)
    assert set(shown) == set(range(len(triggers)))
    for index, text in shown.items():
        assert text.startswith((triggers[index].name or "").strip() or "(unnamed)"), (index, text)
    current = window.trigger_panel.current_trigger_index()
    assert current is None or 0 <= current < len(triggers)


@pytest.mark.parametrize("name,operate", _TRIGGER_OPERATIONS, ids=[n for n, _ in _TRIGGER_OPERATIONS])
def test_a_jump_over_trigger_edits_rebuilds_the_tree_from_the_model(name: str, operate) -> None:
    from test_trigger_undo_viewer import TRIGGER_FIXTURE

    from descape.scenario_io import parse_triggers

    window = _triggers_window(TRIGGER_FIXTURE)
    try:
        start_names = [t.name for t in parse_triggers(window.scenario).triggers]
        operate(window)
        end = window.edit_history.cursor
        assert end == 2, f"{name} did not record two steps"
        end_names = [t.name for t in window.trigger_edits.manager().triggers]
        assert end_names != start_names

        _click_jump(window, 0)
        assert [t.name for t in window.trigger_edits.manager().triggers] == start_names
        _assert_tree_matches_the_model(window)
        if name == "rename":
            assert window.trigger_panel.current_trigger_index() == 1

        _click_jump(window, end)
        assert [t.name for t in window.trigger_edits.manager().triggers] == end_names
        _assert_tree_matches_the_model(window)
        if name == "rename":
            assert window.trigger_panel.current_trigger_index() == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #30 step 15: back to "Opened file", then Save --------------------------


def _paint_in_terrain_mode(window) -> None:
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("draw")
    _paint(window, 7, 7, _TERRAIN_A)


def _unit_edits(window) -> None:
    from PyQt5.QtCore import Qt
    from test_unit_edit_viewer import _REF_ARCHER_P1, _REF_VILLAGER_P2, _select

    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.show_garrisoned_action.setChecked(True)
    window.mode_combo.setCurrentText("Units")
    _select(window, _REF_ARCHER_P1)
    window.on_unit_nudge(1, 0, Qt.NoModifier)
    _select(window, _REF_VILLAGER_P2)
    window.on_unit_delete(Qt.NoModifier)


def _trigger_edits(window) -> None:
    window.mode_combo.setCurrentText("Triggers")
    _rename_twice(window)
    _delete_the_current_and_the_first(window)


def _fixture(name: str):
    from test_trigger_undo_viewer import TRIGGER_FIXTURE
    from test_unit_edit_viewer import FIXTURE_PATH

    return {"units": (FIXTURE_PATH, _unit_edits), "triggers": (TRIGGER_FIXTURE, _trigger_edits)}[name]


@pytest.mark.parametrize("name", ["units", "triggers"])
def test_saving_after_a_jump_to_the_opened_file_writes_the_original_bytes(tmp_path, name: str) -> None:
    from descape.scenario_write import write_scenario

    fixture, edit = _fixture(name)
    copy = tmp_path / fixture.name
    copy.write_bytes(fixture.read_bytes())
    window = conftest.shown_window()
    try:
        window.load_scenario(copy)
        assert window.scenario is not None
        _paint_in_terrain_mode(window)
        edit(window)
        assert window.edit_history.cursor > 2, "every edit recorded its own step"
        # Non-vacuity: the edited state really does write different bytes.
        edited = tmp_path / "edited.aoe2scenario"
        models = {
            "triggers": window.trigger_edits,
            "options": window.option_edits,
            "units": window.unit_edits,
            "messages": window.message_edits,
        }
        write_scenario(window.scenario, edited, **models)
        assert edited.read_bytes() != fixture.read_bytes()

        _click_jump(window, 0)
        assert window.edit_history.cursor == 0
        window.save()

        assert "No changes to save" in window.status_log.toPlainText()
        assert copy.read_bytes() == fixture.read_bytes()
        # No .bak/.orig either: nothing was written. config.yaml is the isolated settings file.
        extra = {p.name for p in tmp_path.iterdir()} - {copy.name, edited.name, "config.yaml"}
        assert not extra
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #30 steps 10-12: terrain-span jumps against a fresh render -------------

_MOVED_REGION = (54, 67, 60, 73)  # where the paste-then-drag block ends up
_INSIDE_MOVED = (55, 68)


def _terrain_span_window(style: str):
    """Blank template, Select tool, shown so a drag can hit a tile."""
    from test_region_select import _select_window

    return _select_window(style)


def _stroke(window, tool: str, tiles, terrain_id: int | None = None) -> None:
    """One recorded stroke over `tiles` with `tool`, then back to Select."""
    window._on_tool_selected(tool)
    if terrain_id is not None:
        window.terrain_panel.set_terrain(terrain_id)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tiles(tiles, 0)
    window.on_edit_stroke_end()
    window._on_tool_selected("select")


def _block(x0: int, y0: int, x1: int, y1: int):
    return [(x, y) for y in range(y0, y1) for x in range(x0, x1)]


def _tile_states(window):
    from descape.edit_history import tile_state

    return [tile_state(t) for t in window.scenario.map_manager.terrain]


def _fresh_render(window, canvas_w: int, canvas_h: int):
    from descape.render import render_terrain_iso_with_proj, render_terrain_sloped_with_proj
    from descape.render_cache import SlopedChunkCache

    sprites = window._cache.sprites_enabled  # priv: the live cache's own sprite flag
    if isinstance(window._cache, SlopedChunkCache):
        full = render_terrain_sloped_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    else:
        full = render_terrain_iso_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    return full[:canvas_h, :canvas_w]


def _move_cursor_over(window, tile):
    """The cursor a plain hover shows: SizeAll only over a movable paste."""
    from test_region_move import _hover

    _hover(window.map_view, tile)
    return window.map_view.cursor().shape()


def _build_terrain_spans(window, snapshots: list) -> dict[str, int]:
    """Paint, elevation, region paste, paste-then-drag and mirror records,
    one each (plus a second paint). Appends the tile states at every cursor
    to `snapshots` and returns each span's cursor."""
    from PyQt5.QtCore import Qt
    from test_region_move import _drag_region

    from descape.mirror_tools import plan_mirror

    history = window.edit_history
    snapshots.append(_tile_states(window))
    cursors = {}

    def record(name: str) -> None:
        cursors[name] = history.cursor
        assert history.cursor == len(snapshots), f"{name} did not record exactly one step"
        snapshots.append(_tile_states(window))

    # Clustered mid-map: a multi-record jump patches one union bbox, so spread-out edits cost seconds.
    _stroke(window, "draw", _block(48, 48, 52, 52), _TERRAIN_A)
    record("paint")
    _stroke(window, "elevation", _block(50, 56, 52, 58))
    record("elevation")
    _stroke(window, "draw", _block(44, 60, 48, 63), _TERRAIN_B)
    record("paint_b")

    window.on_region_selected((48, 54, 54, 60))  # spans the elevation bump and bare ground
    window.copy_region()
    window.on_hover((40, 50))
    window.paste_region()
    record("paste")
    window.on_hover((50, 64))
    window.paste_region()
    _drag_region(window, (50, 64), _MOVED_REGION[:2])
    record("paste_drag")
    assert window._region == _MOVED_REGION, "the drag did not move the pasted block"
    assert _move_cursor_over(window, _INSIDE_MOVED) == Qt.SizeAllCursor, "the moved block is still movable"

    mm = window.scenario.map_manager
    plan = plan_mirror(mm, 1, 0, do_terrain=True, do_elevation=True)
    assert not plan.elevation_violations
    assert window.on_mirror(plan)
    record("mirror")
    return cursors


# A dozen jumps: back and forward, single and multi-step, over every span.
_JUMPS = [0, 6, 3, 1, 5, 2, 4, 0, 5, 3, 6, 2]


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_jumps_over_terrain_spans_repaint_like_a_fresh_render(style: str) -> None:
    import itertools

    import numpy as np
    from PyQt5.QtCore import Qt
    from test_region_move import _drag_region

    from descape.render_cache import IsoChunkCache, SlopedChunkCache
    from descape.viewer import STEPPED_FULL_RERENDER_THRESHOLD

    window = _terrain_span_window(style)
    try:
        expected = {"Stepped": IsoChunkCache, "Sloped": SlopedChunkCache}[style]
        cache = window._cache  # priv: the incremental path must patch this very cache
        assert type(cache) is expected, f"{style} window built a {type(cache).__name__}"
        canvas_w, canvas_h = cache.canvas_dims()

        snapshots: list = []
        cursors = _build_terrain_spans(window, snapshots)
        assert set(_JUMPS) == set(range(len(snapshots)))
        # Every span is small enough for the patch path, not the full re-render fallback.
        for before, after in itertools.pairwise(snapshots):
            changed = sum(a != b for a, b in zip(before, after, strict=True))
            assert 0 < changed < STEPPED_FULL_RERENDER_THRESHOLD
        assert window._cache is cache

        fresh_by_target = {}  # tile states are asserted equal per target, so one render each
        for target in _JUMPS:
            dialog = _open_dialog(window)
            dialog.tree.setCurrentItem(dialog.tree.topLevelItem(target))
            assert dialog.selected_target() == target
            # Re-warm right before the click: other mip levels share the LRU and evict mip-0
            # chunks, and an evicted chunk recomposites fresh, hiding a stale patch.
            cache.render_rect(0, 0, canvas_w, canvas_h)
            dialog.jump_button.click()
            assert window.edit_history.cursor == target
            assert _tile_states(window) == snapshots[target], f"tiles after jumping to {target}"
            assert window._cache is cache, f"jump to {target} rebuilt the cache"
            stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
            if target not in fresh_by_target:
                fresh_by_target[target] = _fresh_render(window, canvas_w, canvas_h)
            assert np.array_equal(stitched, fresh_by_target[target]), f"stale pixels after jumping to {target}"
            # The clipboard is not history: no jump may take Paste away.
            assert window.paste_action.isEnabled()
        assert "re-rendered full map" not in window.status_log.toPlainText()
        assert cursors == {"paint": 1, "elevation": 2, "paint_b": 3, "paste": 4, "paste_drag": 5, "mirror": 6}

        # Step 11: back before the paste, a drag on the old block must not move it.
        last = _JUMPS[-1]
        assert last < cursors["paste"]
        assert window._region == _MOVED_REGION, "a jump does not move the selection"
        assert _move_cursor_over(window, _INSIDE_MOVED) == Qt.CrossCursor, "the gone paste still reads as movable"
        _drag_region(window, _INSIDE_MOVED, (_INSIDE_MOVED[0] + 2, _INSIDE_MOVED[1] + 2))
        assert window.edit_history.cursor == last
        assert len(window.edit_history.records) == len(snapshots) - 1, "the drag pushed a record"
        assert _tile_states(window) == snapshots[last]
    finally:
        window.edit_history.mark_saved()
        window.close()
