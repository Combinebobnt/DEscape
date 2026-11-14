"""Trigger panel: the status column's markers, brushes and re-evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    TRIGGER_FIXTURE,
    _effect_type_id,
    _group,
    _header,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _open,
    _row_widget,
    _sectioned_window,
    _triggers_window,
    _window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- GH #166: trigger status colours -------------------------------------------
#
# The fixture's statuses, measured: triggers 0-2 are complete; trigger 3 is not
# (condition 3 names unit 999, which is not placed; effect 3's area is half-set;
# effect 6, a patrol, has no location). Condition 2 of trigger 3 names unit 502,
# which is placed (player 1).

_OK_HEX = "#4caf50"


_PROBLEM_HEX = "#e05252"


_BROKEN_TRIGGER = 3


_REFERENCED_UNIT_KEY = (1, 502)


def _brush_hex(item, col: int) -> str | None:
    """The row's brush, read back as the default setting hex it is the legible
    shade of (TASK-424), else as drawn. test_trigger_status_contrast.py measures the shade."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPalette

    from descape.viewer_dialogs import legible_status_color

    brush = item.data(col, Qt.ForegroundRole)
    if brush is None:
        return None
    base = item.treeWidget().palette().color(QPalette.Active, QPalette.Base).name()
    shades = {legible_status_color(hex_str, base): hex_str for hex_str in (_OK_HEX, _PROBLEM_HEX)}
    return shades.get(brush.color().name(), brush.color().name())


def _icon_image(item, col: int):
    return item.icon(col).pixmap(16).toImage()


def _expected_icon(status, hex_str: str):
    from descape.trigger_panel_status import status_icon

    return status_icon(status, hex_str).pixmap(16).toImage()


def test_the_panels_own_status_style_defaults_are_the_settings_defaults() -> None:
    from descape import settings
    from descape.trigger_panel_status import StatusStyle

    style = StatusStyle()
    assert style.marker == settings.TRIGGER_STATUS_MARKER_DEFAULT
    assert style.ok == settings.get_default_trigger_status_color("ok")
    assert style.problem == settings.get_default_trigger_status_color("problem")
    assert style.color_ok_rows is True
    assert style.color_sections is settings.get_trigger_status_color_sections() is False


def test_the_trigger_list_has_a_blank_status_column_between_id_and_trigger() -> None:
    """TASK-317 (GH #166 follow-up): four columns, the status one unlabelled
    but explained by its header tooltip."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        header = panel.tree.headerItem()
        assert panel.tree.columnCount() == 4
        assert (panel._COL_POS, panel._COL_ID, panel._COL_STATUS, panel._COL_NAME) == (0, 1, 2, 3)
        assert [header.text(c) for c in range(4)] == [_header(panel), "ID", "", "Trigger"]
        assert "dash" in header.toolTip(panel._COL_STATUS)
    finally:
        window.close()


def test_a_problem_row_has_the_problem_brush_and_a_tooltip_naming_its_reasons() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        col = panel._COL_NAME
        row = panel._item_for_index[_BROKEN_TRIGGER]
        assert _brush_hex(row, col) == _PROBLEM_HEX
        tip = row.toolTip(col).splitlines()
        assert "condition 4 (destroy object): unit 999 (unit object) is not placed on the map" in tip
        assert "effect 4 (kill object): area is only partly set" in tip
        assert "effect 7 (patrol): missing location" in tip
        assert row.text(col) == panel._trigger_label(panel._manager().triggers[_BROKEN_TRIGGER]), (
            "the status leaked into the row text"
        )

        panel.select_trigger(_BROKEN_TRIGGER)
        patrol = panel._entry_item_for("effect", 6)
        assert _brush_hex(patrol, 0) == _PROBLEM_HEX
        assert patrol.toolTip(0) == "missing location"
        assert _brush_hex(panel._entry_item_for("effect", 0), 0) == _OK_HEX
        assert panel._entry_item_for("effect", 0).toolTip(0) == ""
        assert _brush_hex(panel.entry_tree.topLevelItem(0), 0) == _PROBLEM_HEX
        assert _brush_hex(_group(panel, "condition"), 0) == _PROBLEM_HEX
        assert _brush_hex(_group(panel, "effect"), 0) == _PROBLEM_HEX
    finally:
        window.close()


def test_an_ok_row_is_green_and_unbrushed_with_ok_colouring_off() -> None:
    from descape import settings

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        col = panel._COL_NAME
        ok_row = panel._item_for_index[0]
        assert _brush_hex(ok_row, col) == _OK_HEX
        assert ok_row.toolTip(col) == ""

        settings.set_trigger_status_color_ok_rows(False)
        panel.apply_status_style()
        assert _brush_hex(ok_row, col) is None
        assert _brush_hex(panel._item_for_index[_BROKEN_TRIGGER], col) == _PROBLEM_HEX
        panel.select_trigger(0)
        assert _brush_hex(panel.entry_tree.topLevelItem(0), 0) is None
        assert _brush_hex(panel._entry_item_for("effect", 0), 0) is None
    finally:
        window.close()


@pytest.mark.parametrize(
    ("marker", "brushed", "iconed"),
    [("color", True, False), ("icon", False, True), ("both", True, True), ("off", False, False)],
)
def test_each_marker_mode_sets_exactly_its_brush_and_icon(marker: str, brushed: bool, iconed: bool) -> None:
    from descape import settings
    from descape.trigger_status import Status

    settings.set_trigger_status_marker(marker)
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        col, icon_col = panel._COL_NAME, panel._COL_STATUS
        # TASK-317: the icons have their own column, shown only while there are icons.
        assert panel.tree.isColumnHidden(icon_col) is not iconed, marker
        for index, status, hex_str in ((0, Status.OK, _OK_HEX), (_BROKEN_TRIGGER, Status.PROBLEM, _PROBLEM_HEX)):
            row = panel._item_for_index[index]
            assert _brush_hex(row, col) == (hex_str if brushed else None), (marker, index)
            assert _brush_hex(row, icon_col) is None, (marker, index)
            assert row.icon(col).isNull(), (marker, index, "an icon on the name column")
            assert row.icon(icon_col).isNull() is not iconed, (marker, index)
            if iconed:
                assert _icon_image(row, icon_col) == _expected_icon(status, hex_str), (marker, index)
        # The tooltip is not a marker: every mode, "off" included, explains a problem row.
        broken = panel._item_for_index[_BROKEN_TRIGGER]
        assert broken.toolTip(col) and broken.toolTip(icon_col) == broken.toolTip(col)
    finally:
        window.close()


@pytest.mark.parametrize("marker", ["color", "icon", "both", "off"])
def test_with_marking_passes_off_an_ok_row_has_no_marker_in_any_mode(marker: str) -> None:
    """User decision, 2026-10-07: the setting means "mark triggers that pass",
    so off drops the tick as well as the green, and only problems are marked."""
    from descape import settings
    from descape.trigger_status import Status

    settings.set_trigger_status_marker(marker)
    settings.set_trigger_status_color_ok_rows(False)
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        col, icon_col = panel._COL_NAME, panel._COL_STATUS
        panel.select_trigger(_BROKEN_TRIGGER)
        style = panel._status_style
        for ok_row, ok_col, ok_icon_col in (
            (panel._item_for_index[0], col, icon_col),
            (panel._entry_item_for("effect", 0), 0, 0),
        ):
            assert _brush_hex(ok_row, ok_col) is None, marker
            assert ok_row.icon(ok_icon_col).isNull(), marker
        for problem_row, problem_col, problem_icon_col in (
            (panel._item_for_index[_BROKEN_TRIGGER], col, icon_col),
            (panel._entry_item_for("effect", 6), 0, 0),
        ):
            assert _brush_hex(problem_row, problem_col) == (_PROBLEM_HEX if style.colours else None), marker
            assert problem_row.icon(problem_icon_col).isNull() is not style.icons, marker
            if style.icons:
                assert _icon_image(problem_row, problem_icon_col) == _expected_icon(Status.PROBLEM, _PROBLEM_HEX)
        assert panel._item_for_index[_BROKEN_TRIGGER].icon(col).isNull(), "an icon on the name column"
    finally:
        window.close()


@pytest.mark.parametrize("flag", ["display_as_objective", "display_on_screen"])
def test_an_objective_or_on_screen_trigger_with_no_effects_is_not_a_problem(flag: str) -> None:
    """User decisions, 2026-10-07; the Effects group row follows the trigger row."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        col = panel._COL_NAME
        trigger = panel._manager().triggers[0]
        trigger.display_as_objective, trigger.header, trigger.display_on_screen = 0, 0, 0
        setattr(trigger, flag, 1)  # so this flag alone exempts it
        panel.select_trigger(0)  # "Fixture: setup", one effect
        window.entry_structural_edit("delete", 0, "effect", 0, -1)
        row = panel._item_for_index[0]
        assert _brush_hex(row, col) == _OK_HEX
        assert row.toolTip(col) == ""
        assert _brush_hex(_group(panel, "effect"), 0) == _OK_HEX
        assert _group(panel, "effect").toolTip(0) == ""
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_fixing_a_field_through_the_form_flips_its_row_trigger_root_and_group() -> None:
    """The _refresh_labels() path: activate_trigger requires a trigger_id."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        col = panel._COL_NAME
        panel.select_trigger(2)  # "Fixture: references"
        panel.entry_tree.setCurrentItem(panel._entry_item_for("effect", 0))
        widget = _row_widget(panel, "trigger_id")
        widget.setCurrentIndex(widget.findData(-1))

        row = panel._item_for_index[2]
        assert _brush_hex(row, col) == _PROBLEM_HEX
        assert "effect 1 (activate trigger): missing trigger id" in row.toolTip(col).splitlines()
        assert _brush_hex(panel._entry_item_for("effect", 0), 0) == _PROBLEM_HEX
        assert _brush_hex(panel.entry_tree.topLevelItem(0), 0) == _PROBLEM_HEX
        assert _brush_hex(_group(panel, "effect"), 0) == _PROBLEM_HEX
        assert _brush_hex(_group(panel, "condition"), 0) == _OK_HEX

        widget.setCurrentIndex(widget.findData(1))
        for item, item_col in (
            (row, col),
            (panel._entry_item_for("effect", 0), 0),
            (panel.entry_tree.topLevelItem(0), 0),
            (_group(panel, "effect"), 0),
        ):
            assert _brush_hex(item, item_col) == _OK_HEX
            assert item.toolTip(item_col) == ""
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_deleting_the_last_effect_turns_the_trigger_red_and_adding_one_turns_it_green() -> None:
    """The refresh_entries() path, which never reaches show_scenario()."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        col = panel._COL_NAME
        # Shown on screen, which also exempts it (user decision, 2026-10-07).
        panel._manager().triggers[0].display_on_screen = 0
        panel.select_trigger(0)  # "Fixture: setup", one effect
        window.entry_structural_edit("delete", 0, "effect", 0, -1)
        row = panel._item_for_index[0]
        assert _brush_hex(row, col) == _PROBLEM_HEX
        assert row.toolTip(col) == "has no effects"
        assert _brush_hex(_group(panel, "effect"), 0) == _PROBLEM_HEX
        assert _group(panel, "effect").toolTip(0) == "has no effects"

        window.entry_structural_edit("new", 0, "effect", -1, _effect_type_id(panel, "send_chat"))
        assert _brush_hex(row, col) == _OK_HEX
        assert _brush_hex(_group(panel, "effect"), 0) == _OK_HEX
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_retyping_an_entry_flips_its_status() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_BROKEN_TRIGGER)
        assert _brush_hex(panel._entry_item_for("effect", 6), 0) == _PROBLEM_HEX  # patrol, no location
        window.entry_structural_edit("retype", _BROKEN_TRIGGER, "effect", 6, _effect_type_id(panel, "send_chat"))
        assert _brush_hex(panel._entry_item_for("effect", 6), 0) == _OK_HEX
        tip = panel._item_for_index[_BROKEN_TRIGGER].toolTip(panel._COL_NAME)
        assert "patrol" not in tip and "effect 4 (kill object)" in tip
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_setting_an_area_through_the_tile_pick_turns_the_row_ok() -> None:
    """The set_entry_field_group() path (GH #138's Set Area)."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_BROKEN_TRIGGER)
        panel.entry_tree.setCurrentItem(panel._entry_item_for("effect", 3))  # kill_object, half-set area
        assert _brush_hex(panel._entry_item_for("effect", 3), 0) == _PROBLEM_HEX
        panel._group_rows["area"].set_button.click()
        assert window._tile_picker is not None
        window._on_rect_picked(10, 12, 20, 22)

        assert _brush_hex(panel._entry_item_for("effect", 3), 0) == _OK_HEX
        tip = panel._item_for_index[_BROKEN_TRIGGER].toolTip(panel._COL_NAME)
        assert "kill object" not in tip and "effect 7 (patrol)" in tip
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_unit_delete_restyles_in_triggers_mode_but_waits_for_mode_entry_elsewhere() -> None:
    """The refresh_unit_reference_labels() path and its Triggers-mode gate."""
    from descape.trigger_status import Status

    window = _triggers_window()
    try:
        panel = window.trigger_panel

        def destroy_status():
            return panel.status_of(_BROKEN_TRIGGER).conditions[2].status

        assert destroy_status() is Status.OK  # destroy_object on unit 502
        assert _BROKEN_TRIGGER in panel._unit_ref_triggers

        window.mode_combo.setCurrentText("Units")
        removed, _cascaded, _blocked = window._find_delete_keys([_REFERENCED_UNIT_KEY])
        assert removed == 1
        assert destroy_status() is Status.OK, "a Units-mode unit edit recomputed trigger statuses"
        window.mode_combo.setCurrentText("Triggers")
        assert destroy_status() is Status.PROBLEM, "re-entering Triggers mode did not recompute"

        window.undo()  # the unit is back, in Triggers mode
        assert destroy_status() is Status.OK
        panel.select_trigger(_BROKEN_TRIGGER)
        destroy_row = panel._entry_item_for("condition", 2)
        assert _brush_hex(destroy_row, 0) == _OK_HEX

        window.redo()  # deleted again, in Triggers mode
        assert destroy_status() is Status.PROBLEM
        destroy_row = panel._entry_item_for("condition", 2)
        assert _brush_hex(destroy_row, 0) == _PROBLEM_HEX
        assert destroy_row.toolTip(0) == "unit 502 (unit object) is not placed on the map"
        assert _brush_hex(panel._item_for_index[_BROKEN_TRIGGER], panel._COL_NAME) == _PROBLEM_HEX
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_clear_document_resets_the_status_cache() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert panel.status_of(_BROKEN_TRIGGER) is not None
        panel.clear_document()
        assert panel.status_of(_BROKEN_TRIGGER) is None
        assert panel._unit_ref_triggers == set()
    finally:
        window.close()


def test_a_settings_change_restyles_the_rows_live() -> None:
    from descape import settings
    from descape.settings_dialog import SettingsDialog
    from descape.trigger_status import Status

    window = _triggers_window()
    dialog = SettingsDialog(window)
    try:
        panel = window.trigger_panel
        col, icon_col = panel._COL_NAME, panel._COL_STATUS
        broken, ok = panel._item_for_index[_BROKEN_TRIGGER], panel._item_for_index[0]
        panel.select_trigger(_BROKEN_TRIGGER)
        assert panel.tree.isColumnHidden(icon_col), "the default marker, Colour, shows no status column"

        combo = dialog.trigger_status_marker_combo
        combo.setCurrentIndex(combo.findData("icon"))
        assert settings.get_trigger_status_marker() == "icon"
        assert not panel.tree.isColumnHidden(icon_col)
        assert _brush_hex(broken, col) is None and not broken.icon(icon_col).isNull()
        assert broken.icon(col).isNull()
        assert not panel._entry_item_for("effect", 6).icon(0).isNull()

        combo.setCurrentIndex(combo.findData("both"))
        assert dialog.trigger_status_ok_rows_check.text() == "Mark triggers that pass"
        dialog.trigger_status_ok_rows_check.setChecked(False)
        assert _brush_hex(ok, col) is None and ok.icon(icon_col).isNull()
        assert _icon_image(broken, icon_col) == _expected_icon(Status.PROBLEM, _PROBLEM_HEX)
        dialog.trigger_status_ok_rows_check.setChecked(True)
        assert _icon_image(ok, icon_col) == _expected_icon(Status.OK, _OK_HEX)

        dialog._apply_trigger_status_color("problem", "#123456")
        assert settings.get_trigger_status_color("problem") == "#123456"
        assert _brush_hex(broken, col) == "#123456"
        assert _icon_image(broken, icon_col) == _expected_icon(Status.PROBLEM, "#123456")
        assert _brush_hex(panel._entry_item_for("effect", 6), 0) == "#123456"

        combo.setCurrentIndex(combo.findData("off"))
        assert panel.tree.isColumnHidden(icon_col) and broken.icon(icon_col).isNull()
    finally:
        dialog.close()
        window.close()


_EMPTY_HEX = "#8a8a8a"


def test_an_empty_section_shows_a_dash_and_a_complete_one_a_tick_with_no_brush() -> None:
    """TASK-317 (GH #166 follow-up): a section row is marked by its own
    entries, a dash when it has none, and stays uncoloured by default."""
    from PyQt5.QtCore import Qt

    from descape import settings
    from descape.trigger_status import Status

    settings.set_trigger_status_marker("both")
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        col, icon_col = panel._COL_NAME, panel._COL_STATUS
        empty, complete = panel._item_for_index[4], panel._item_for_index[2]
        assert panel.status_of(4).trigger.status is Status.EMPTY and panel.status_of(4).divider
        dash = _expected_icon(Status.EMPTY, _EMPTY_HEX)
        assert _icon_image(empty, icon_col) == dash
        assert dash != _expected_icon(Status.OK, _EMPTY_HEX) and dash != _expected_icon(Status.PROBLEM, _EMPTY_HEX)
        assert _brush_hex(empty, col) is None and empty.icon(col).isNull()
        assert empty.toolTip(icon_col) == empty.toolTip(col) == "Section with no conditions or effects"
        assert _icon_image(complete, icon_col) == _expected_icon(Status.OK, _OK_HEX)
        assert _brush_hex(complete, col) is None and complete.toolTip(col) == ""
        assert _brush_hex(panel._item_for_index[3], col) == _PROBLEM_HEX, "a member row keeps its own colour"

        synthetic = panel.tree.topLevelItem(0)
        assert synthetic.data(panel._ROLE_COL, Qt.UserRole) is None, "fixture assumption"
        assert all(synthetic.icon(c).isNull() and synthetic.toolTip(c) == "" for c in range(4))
        assert _brush_hex(synthetic, col) is None

        panel.select_trigger(4)
        root = panel.entry_tree.topLevelItem(0)
        assert _icon_image(root, 0) == dash and _brush_hex(root, 0) is None

        # Colour section rows never colours a dash.
        settings.set_trigger_status_color_sections(True)
        panel.apply_status_style()
        assert _brush_hex(empty, col) is None and _brush_hex(complete, col) == _OK_HEX

        # Passes off: "only problems stand out", so the dash goes with the tick.
        settings.set_trigger_status_color_ok_rows(False)
        panel.apply_status_style()
        assert empty.icon(icon_col).isNull() and complete.icon(icon_col).isNull()
        assert panel.entry_tree.topLevelItem(0).icon(0).isNull()
        assert not panel._item_for_index[3].icon(icon_col).isNull()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_section_with_a_broken_condition_has_the_cross_and_a_brush_only_on_request() -> None:
    from descape import settings
    from descape.settings_dialog import SettingsDialog
    from descape.trigger_status import Status

    settings.set_trigger_status_marker("both")
    window = _triggers_window()
    dialog = None
    try:
        panel = window.trigger_panel
        col, icon_col = panel._COL_NAME, panel._COL_STATUS
        panel.select_trigger(_BROKEN_TRIGGER)
        name = _row_widget(panel, "name")
        name.setText("--- Section ---")
        name.editingFinished.emit()  # crosses is_divider: a full repopulate

        divider = panel._item_for_index[_BROKEN_TRIGGER]
        assert panel.status_of(_BROKEN_TRIGGER).divider
        assert _icon_image(divider, icon_col) == _expected_icon(Status.PROBLEM, _PROBLEM_HEX)
        tip = divider.toolTip(col).splitlines()
        assert "condition 4 (destroy object): unit 999 (unit object) is not placed on the map" in tip
        assert divider.toolTip(icon_col) == divider.toolTip(col)
        assert _brush_hex(divider, col) is None
        panel.select_trigger(_BROKEN_TRIGGER)
        assert _brush_hex(panel.entry_tree.topLevelItem(0), 0) is None
        assert _icon_image(panel.entry_tree.topLevelItem(0), 0) == _expected_icon(Status.PROBLEM, _PROBLEM_HEX)
        assert _brush_hex(_group(panel, "condition"), 0) == _PROBLEM_HEX, "the group rows keep their colour"
        assert _brush_hex(panel._entry_item_for("effect", 6), 0) == _PROBLEM_HEX

        dialog = SettingsDialog(window)
        check = dialog.trigger_status_sections_check
        assert check.text() == "Colour section rows" and not check.isChecked()
        check.setChecked(True)
        assert settings.get_trigger_status_color_sections() is True
        assert _brush_hex(divider, col) == _PROBLEM_HEX
        assert _brush_hex(panel.entry_tree.topLevelItem(0), 0) == _PROBLEM_HEX
        assert _brush_hex(panel._item_for_index[0], col) == _OK_HEX, "an ordinary row is unaffected"
        check.setChecked(False)
        assert _brush_hex(divider, col) is None
        assert _brush_hex(panel.entry_tree.topLevelItem(0), 0) is None
    finally:
        if dialog is not None:
            dialog.close()
        window.edit_history.mark_saved()
        window.close()


def test_the_filter_still_matches_on_the_name_not_the_status() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.filter_edit.setText("variable")
        assert not panel._item_for_index[_BROKEN_TRIGGER].isHidden()
        assert panel._item_for_index[0].isHidden()
        panel.filter_edit.setText("missing location")
        assert panel._item_for_index[_BROKEN_TRIGGER].isHidden(), "the filter matched a status reason"
    finally:
        window.close()


def test_a_live_marker_change_sizes_the_status_and_name_columns_like_a_fresh_populate() -> None:
    """An icon shows the trigger list's status column (TASK-317) and widens
    the detail tree's Item column. A live restyle batches its setData calls
    by switching the trigger tree's columns off ResizeToContents, so they must
    come back, and both trees must end up as wide as a repopulate makes them.
    The trigger names keep their width: the icon is not in their column."""
    from PyQt5.QtWidgets import QApplication, QHeaderView

    from descape import settings

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_BROKEN_TRIGGER)
        QApplication.processEvents()
        header = panel.tree.header()

        def widths():
            return (
                header.sectionSize(panel._COL_STATUS),
                header.sectionSize(panel._COL_NAME),
                panel.entry_tree.columnWidth(0),
            )

        before = widths()
        assert before[0] == 0, "the default marker hides the status column"

        settings.set_trigger_status_marker("icon")
        panel.apply_status_style()
        QApplication.processEvents()
        assert all(header.sectionResizeMode(c) == QHeaderView.ResizeToContents for c in range(header.count()))
        live = widths()

        window._show_triggers()
        QApplication.processEvents()
        fresh = widths()
        assert live == fresh
        assert live[0] >= 16, "the status column did not open to fit its icon"
        assert live[1] == before[1], "the icon moved into the name column"
        assert live[2] > before[2], "the icon did not widen the detail tree's Item column"
    finally:
        window.close()


def _count_evaluations(monkeypatch) -> list:
    """Every trigger object evaluate_with_references() is called on, in call
    order: the panel's pass, and what evaluate_trigger() delegates to."""
    from descape import trigger_status

    real = trigger_status.evaluate_with_references
    calls: list = []

    def counting(trigger, context):
        calls.append(trigger)
        return real(trigger, context)

    monkeypatch.setattr(trigger_status, "evaluate_with_references", counting)
    return calls


def _evaluated_indices(panel, calls) -> set[int]:
    triggers = panel._manager().triggers
    return {i for i, t in enumerate(triggers) if any(c is t for c in calls)}


def test_a_repopulate_reevaluates_only_the_edited_trigger_and_undo_restores_its_status(monkeypatch) -> None:
    """The status memo across same-document show_scenario() calls: a rename
    that crosses is_divider is a refresh="panel" edit of one trigger in place."""
    from descape.trigger_status import Status

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        col = panel._COL_NAME
        before = panel.status_of(_BROKEN_TRIGGER)
        assert before.trigger.status is Status.PROBLEM
        calls = _count_evaluations(monkeypatch)

        window._show_triggers()
        assert calls == [], "an unedited repopulate re-evaluated a trigger"

        panel.select_trigger(_BROKEN_TRIGGER)
        name = _row_widget(panel, "name")
        name.setText("--- Section ---")
        name.editingFinished.emit()
        assert panel.status_of(_BROKEN_TRIGGER).divider
        assert panel.status_of(_BROKEN_TRIGGER).trigger == before.trigger, "a section rolls up its own entries"
        assert _evaluated_indices(panel, calls) == {_BROKEN_TRIGGER}
        assert len(calls) == sum(1 for c in calls if c is panel._manager().triggers[_BROKEN_TRIGGER])

        calls.clear()
        window.undo()
        assert panel.status_of(_BROKEN_TRIGGER) == before
        assert _brush_hex(panel._item_for_index[_BROKEN_TRIGGER], col) == _PROBLEM_HEX
        assert _brush_hex(panel._item_for_index[0], col) == _OK_HEX
        assert _evaluated_indices(panel, calls) == {_BROKEN_TRIGGER}, "undo re-evaluated an untouched trigger"

        calls.clear()
        window.redo()
        assert panel.status_of(_BROKEN_TRIGGER).divider
        assert _evaluated_indices(panel, calls) == {_BROKEN_TRIGGER}
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_trigger_count_change_reevaluates_an_unedited_trigger_naming_a_trigger_id(tmp_path, monkeypatch) -> None:
    """Trigger 2 activates trigger 4, which does not exist until New appends it.
    Trigger 2 is never edited, so only the count can make its memo stale."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel
    from descape.trigger_status import Status

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = TriggerEditModel(loaded)
    model.manager().triggers[2].effects[0].trigger_id = 4
    model.mark_dirty(2)
    path = tmp_path / "dangling.aoe2scenario"
    write_scenario(loaded, path, triggers=model)

    window = _open(path)
    try:
        panel = window.trigger_panel
        dangling = "effect 1 (activate trigger): trigger 4 does not exist"
        assert dangling in panel.status_of(2).trigger.reasons
        calls = _count_evaluations(monkeypatch)

        window.trigger_structural_edit("new", [])
        assert len(panel._manager().triggers) == 5
        assert panel.status_of(2).trigger.status is Status.OK
        assert _evaluated_indices(panel, calls) == {2, 4}, "only the new trigger and the one naming an id"

        calls.clear()
        window.undo()
        assert dangling in panel.status_of(2).trigger.reasons
        assert _evaluated_indices(panel, calls) == {2}
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_trigger_status_restyles_a_590_trigger_file_without_a_resize_per_row() -> None:
    """Measured 2026-10-07: 2.3 s for a live marker change and 453 ms for a
    Triggers-mode unit edit on old-allies-final-v2 when every setData re-sized
    the content-sized columns; 3 ms and 8 ms batched. 250 ms is far from both."""
    path = Path(__file__).resolve().parent.parent / "examples" / "old-allies-final-v2.aoe2scenario"
    if not path.is_file():
        pytest.skip(f"missing corpus file: {path}")
    _assert_status_restyle_is_batched(path)


@pytest.mark.test_map
def test_trigger_status_restyles_the_600_trigger_test_map_without_a_resize_per_row() -> None:
    """The same 250 ms gate on the generated test map, in the quick suite."""
    _assert_status_restyle_is_batched(conftest.TEST_MAP_PATH)


def _assert_status_restyle_is_batched(path: Path) -> None:
    import time

    from PyQt5.QtWidgets import QApplication

    from descape import settings

    window = _window()
    try:
        window.load_scenario(path)
        window.mode_combo.setCurrentText("Triggers")
        QApplication.processEvents()
        panel = window.trigger_panel
        assert len(panel._item_for_index) > 500

        settings.set_trigger_status_marker("both")
        start = time.perf_counter()
        panel.apply_status_style()
        restyle_ms = (time.perf_counter() - start) * 1e3

        assert len(panel._unit_ref_triggers) > 50
        start = time.perf_counter()
        panel.refresh_unit_reference_labels(recompute_status=True)
        unit_ms = (time.perf_counter() - start) * 1e3
        assert restyle_ms < 250, f"apply_status_style took {restyle_ms:.0f} ms"
        assert unit_ms < 250, f"refresh_unit_reference_labels took {unit_ms:.0f} ms"
    finally:
        window.edit_history.mark_saved()
        window.close()
