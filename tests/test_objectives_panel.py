"""GH #391: the Objectives dock. Its writes are one undo step each, its rows
follow game order and the Triggers panel's selection, and its preview never
touches the scenario."""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    TRIGGER_FIXTURE,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _window,
)

from descape import settings
from descape.viewer_toolbar import WINDOW_STATE_VERSION

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# (description_order, header, objective, on screen, enabled, short description) per fixture trigger; 3 stays plain.
_BAKED = (
    (100, 1, 1, 1, 1, "Main Objective(s)"),
    (99, 0, 1, 1, 1, "+ Destroy the Invaders"),
    (50, 0, 1, 1, 0, "<RED>- Defend the Castle"),
)


def _settle() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    QApplication.processEvents()


def _objectives_file(tmp_path: Path) -> Path:
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = TriggerEditModel(loaded)
    triggers = model.manager().triggers
    for index, (order, header, objective, screen, enabled, short) in enumerate(_BAKED):
        trigger = triggers[index]
        trigger.description_order = order
        trigger.header = header
        trigger.display_as_objective = objective
        trigger.display_on_screen = screen
        trigger.enabled = enabled
        trigger.short_description = short
        model.mark_dirty(index)
    path = tmp_path / "objectives.aoe2scenario"
    write_scenario(loaded, path, triggers=model)
    return path


def _open(tmp_path: Path):
    window = _window()
    window.load_scenario(_objectives_file(tmp_path))
    window.mode_combo.setCurrentText("Triggers")
    _settle()
    return window


@pytest.fixture
def window(tmp_path):
    window = _open(tmp_path)
    yield window
    window._confirm_discard_changes = lambda: True
    conftest.close_window(window)


def _order(window) -> list[int]:
    return [row.index for row in window.objectives_panel.rows()]


def _triggers(window):
    from descape.scenario_io import parse_triggers

    return parse_triggers(window.scenario).triggers


def _records(window) -> int:
    return window.edit_history.cursor


def _item(window, index: int):
    tree = window.objectives_panel.tree
    for position in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(position)
        if item.data(0, 0x0100) == index:
            return item
    raise AssertionError(f"no row for trigger {index}")


def _front_tab(window, dock) -> str:
    from PyQt5.QtWidgets import QTabBar

    for bar in window.findChildren(QTabBar):
        titles = [bar.tabText(i) for i in range(bar.count())]
        if bar.isVisible() and dock.windowTitle() in titles:
            return bar.tabText(bar.currentIndex())
    raise AssertionError(f"{dock.windowTitle()} is not in a tab group")


# -- rows -----------------------------------------------------------------------


def test_rows_list_objectives_in_game_order_and_grey_disabled_triggers(window):
    from descape.objectives_panel import COL_TRIGGER, DISABLED_TIP

    panel = window.objectives_panel
    assert _order(window) == [0, 1, 2]
    assert panel.editable()
    assert _item(window, 2).toolTip(COL_TRIGGER) == DISABLED_TIP
    assert [entry.text for entry in panel.box.entries()] == [short for *_, short in _BAKED]


# -- writes ---------------------------------------------------------------------


def test_a_flag_set_on_two_rows_is_one_undo_step(window):
    from PyQt5.QtCore import Qt

    from descape.objectives_panel import COL_MUTE

    panel = window.objectives_panel
    panel.select_indices([1, 2])
    before = _records(window)
    _item(window, 1).setCheckState(COL_MUTE, Qt.Checked)
    _settle()
    assert _records(window) == before + 1
    assert [bool(_triggers(window)[i].mute_objectives) for i in (1, 2)] == [True, True]
    assert [row.mute for row in panel.rows()] == [False, True, True]
    window.undo()
    assert [bool(_triggers(window)[i].mute_objectives) for i in (1, 2)] == [False, False]
    assert not any(row.mute for row in panel.rows()), "undo refreshes the list"
    window.redo()
    assert [row.mute for row in panel.rows()] == [False, True, True], "redo refreshes the list"


def test_moving_a_row_up_rewrites_its_order_and_survives_a_save(window, tmp_path):
    panel = window.objectives_panel
    panel.select_indices([2])
    before = _records(window)
    panel.up_button.click()
    assert _order(window) == [0, 2, 1]
    assert _records(window) == before + 1
    window.undo()
    assert _order(window) == [0, 1, 2]
    assert _triggers(window)[2].description_order == 50
    window.redo()
    window.save()
    reopened = _window()
    try:
        reopened.load_scenario(window.scenario.path)
        reopened.mode_combo.setCurrentText("Triggers")
        _settle()
        assert _order(reopened) == [0, 2, 1]
    finally:
        conftest.close_window(reopened)


def test_a_drop_with_no_gap_shifts_the_row_below_down(window):
    panel = window.objectives_panel
    panel.drop_rows([2], 1)
    _settle()
    assert _order(window) == [0, 2, 1]
    assert [t.description_order for t in _triggers(window)[:3]] == [100, 98, 99]


def test_a_drop_into_a_gap_changes_only_the_moved_row(window):
    panel = window.objectives_panel
    panel.drop_rows([0], 3)
    _settle()
    assert _order(window) == [1, 2, 0]
    assert [t.description_order for t in _triggers(window)[:3]] == [24, 99, 50]


def test_inline_short_description_and_order_edits_are_one_step_each(window):
    from PyQt5.QtWidgets import QLineEdit

    from descape.objectives_panel import COL_ORDER, COL_SHORT

    tree = window.objectives_panel.tree
    for column, kind, value, attr in (
        (COL_SHORT, QLineEdit, "Hold the bridge", "short_description"),
        (COL_ORDER, QLineEdit, 75, "description_order"),
    ):
        before = _records(window)
        item = _item(window, 1)
        tree.editItem(item, column)
        editor = tree.indexWidget(tree.indexFromItem(item, column))
        assert isinstance(editor, kind)
        editor.setText(str(value))
        tree.commitData(editor)
        _settle()
        assert _records(window) == before + 1, attr
        assert getattr(_triggers(window)[1], attr) == value


def test_an_inline_commit_that_changes_nothing_records_nothing(window):
    from descape.objectives_panel import COL_ORDER, COL_SHORT

    tree = window.objectives_panel.tree
    before = _records(window)
    for column in (COL_SHORT, COL_ORDER):
        item = _item(window, 1)
        tree.editItem(item, column)
        tree.commitData(tree.indexWidget(tree.indexFromItem(item, column)))
        _settle()
    assert _records(window) == before and not window.edit_history.is_dirty


def test_an_order_past_the_signed_32_bit_limit_is_refused(window):
    from descape.objectives_panel import COL_ORDER

    window._objective_order({1: 2**31 - 1}, "Set objective order")
    tree = window.objectives_panel.tree
    item = _item(window, 1)
    tree.editItem(item, COL_ORDER)
    editor = tree.indexWidget(tree.indexFromItem(item, COL_ORDER))
    assert editor.text() == str(2**31 - 1)
    editor.setText("2147483648")
    tree.commitData(editor)
    _settle()
    assert _triggers(window)[1].description_order == 2**31 - 1, "past 2**31 - 1 is refused"


def test_new_objective_and_new_header_are_one_step_each(window):
    panel = window.objectives_panel
    count = len(_triggers(window))
    for kind, header in (("objective", False), ("header", True)):
        panel.select_index(1)
        before = _records(window)
        window._objective_new(kind, panel.current_index())
        assert _records(window) == before + 1
        added = _triggers(window)[-1]
        assert bool(added.header) == header
        assert added.display_as_objective and added.display_on_screen
        assert added.short_description.startswith("+ ") != header, "an objective gets a checkbox, a header none"
        position = _order(window).index(len(_triggers(window)) - 1)
        assert position == 2, "placed under the selected row"
    assert len(_triggers(window)) == count + 2
    orders = [row.order for row in panel.rows()]
    assert orders == sorted(orders, reverse=True) and min(orders) >= 0
    window.undo()
    window.undo()
    assert len(_triggers(window)) == count


def test_add_trigger_promotes_and_remove_clears_but_keeps_the_trigger(window, monkeypatch):
    panel = window.objectives_panel
    assert [index for index, _label in window._objective_candidates()] == [3]
    monkeypatch.setattr(panel, "pick_triggers", lambda candidates: [3])
    panel.select_index(0)
    panel.add_trigger_button.click()
    assert _order(window) == [0, 3, 1, 2]
    trigger = _triggers(window)[3]
    assert trigger.display_as_objective and trigger.display_on_screen
    count = len(_triggers(window))
    before = _records(window)
    panel.select_indices([3, 2])
    panel.remove_button.click()
    assert _records(window) == before + 1
    assert _order(window) == [0, 1]
    assert len(_triggers(window)) == count
    assert not _triggers(window)[3].display_as_objective


def test_a_row_selects_its_trigger_and_back(window):
    panel = window.objectives_panel
    panel.tree.setCurrentItem(_item(window, 2))
    _settle()
    assert window.trigger_panel.current_trigger_index() == 2
    window.trigger_panel.select_triggers([1])
    _settle()
    assert panel.selected_indices() == [1]
    window.trigger_panel.select_triggers([3])
    _settle()
    assert panel.selected_indices() == [], "a trigger with no row clears the selection"


def test_a_lock_or_an_unwritable_section_greys_editing_and_keeps_the_preview(window):
    from descape import game_dataset

    panel = window.objectives_panel
    window.set_game_data(game_dataset.ROR)
    try:
        assert not panel.editable()
        assert not panel.new_objective_button.isEnabled() and not panel.renumber_button.isEnabled()
        assert panel.note_label.text() == window.game_data_lock_reason()
        assert len(panel.box.entries()) == 3, "the preview still draws"
    finally:
        window.set_game_data(settings.GAME_DATA_DEFAULT)
    assert panel.editable()
    window.scenario.trigger_write_supported = False
    window.refresh_objectives()
    assert not panel.editable() and not panel.add_trigger_button.isEnabled()


# -- preview --------------------------------------------------------------------


def test_preview_place_combo_and_view_menu_stay_in_step_and_persist(window):
    from descape.objectives_panel import PLACE_DOCK, PLACE_MAP, PLACE_OFF

    panel = window.objectives_panel
    overlay = window.objectives_overlay
    assert not overlay.isVisible() and panel.preview.isVisible()
    panel.place_combo.setCurrentIndex(1)
    _settle()
    assert settings.get_objectives_preview() == PLACE_MAP
    assert window.objectives_overlay_action.isChecked()
    assert overlay.isVisible() and not panel.preview.isVisible()
    window.mode_combo.setCurrentText("View")
    _settle()
    assert not overlay.isVisible(), "Triggers mode only"
    window.mode_combo.setCurrentText("Triggers")
    _settle()
    assert overlay.isVisible()
    window.objectives_overlay_action.setChecked(False)
    assert panel.preview_place() == PLACE_DOCK and not overlay.isVisible()
    assert settings.get_objectives_preview() == PLACE_DOCK
    panel.place_combo.setCurrentIndex(2)
    assert settings.get_objectives_preview() == PLACE_OFF
    assert not overlay.isVisible() and not panel.preview.isVisible()


def test_at_game_start_drops_disabled_objectives_and_persists(window):
    panel = window.objectives_panel
    panel.start_radio.setChecked(True)
    assert [entry.index for entry in panel.box.entries()] == [0, 1]
    assert settings.get_objectives_at_start() is True


def test_ticks_change_the_box_only_and_clear_on_another_document(window, tmp_path):
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest

    panel = window.objectives_panel
    before = _records(window)
    (index, rect), *_ = panel.box.check_rects()
    QTest.mouseClick(panel.box, Qt.LeftButton, pos=QPoint(round(rect.center().x()), round(rect.center().y())))
    assert panel.ticked() == {index} and panel.box.state.ticked == {index}
    assert _records(window) == before and not window.edit_history.is_dirty
    (tmp_path / "again").mkdir()
    window.load_scenario(_objectives_file(tmp_path / "again"))
    _settle()
    assert panel.ticked() == set()


def test_the_overlay_takes_clicks_on_its_checkboxes_only(window):
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest

    panel = window.objectives_panel
    panel.place_combo.setCurrentIndex(1)
    _settle()
    overlay = window.objectives_overlay
    (index, rect), *_ = overlay.check_rects()
    hits = overlay._hits
    assert hits.isVisible()
    QTest.mouseClick(hits, Qt.LeftButton, pos=QPoint(round(rect.center().x()), round(rect.center().y())))
    assert panel.ticked() == {index}
    assert not hits.mask().contains(QPoint(overlay.width() - 2, 2)), "the title row lets the map have the mouse"


# -- dock -----------------------------------------------------------------------


def test_the_dock_is_tabbed_behind_the_status_log_and_triggers_mode_only(window):
    dock = window.objectives_dock
    action = dock.toggleViewAction()
    assert window.log_dock in window.tabifiedDockWidgets(dock)
    assert _front_tab(window, window.log_dock) == "Status Log"
    assert not dock.isHidden() and action.isEnabled()
    for mode in ("View", "Units", "Terrain"):
        window.mode_combo.setCurrentText(mode)
        _settle()
        assert dock.isHidden() and not action.isEnabled(), mode
    window.mode_combo.setCurrentText("Triggers")
    _settle()
    assert not dock.isHidden()
    assert _front_tab(window, window.log_dock) == "Status Log", "not raised over the log"


def test_a_dock_closed_in_triggers_mode_stays_closed_until_reopened(window):
    dock = window.objectives_dock
    dock.toggleViewAction().trigger()
    _settle()
    assert dock.isHidden()
    window.mode_combo.setCurrentText("Units")
    _settle()
    window.mode_combo.setCurrentText("Triggers")
    _settle()
    assert dock.isHidden()
    dock.toggleViewAction().trigger()
    window.mode_combo.setCurrentText("Units")
    _settle()
    window.mode_combo.setCurrentText("Triggers")
    _settle()
    assert not dock.isHidden()


def _front(window, dock) -> None:
    dock.raise_()
    for _ in range(3):
        _settle()


def test_the_objectives_tab_grows_its_row_to_fit_and_the_log_tab_puts_it_back(window):
    from PyQt5.QtCore import Qt

    dock, panel, log = window.objectives_dock, window.objectives_panel, window.log_dock
    map_area = window.centralWidget()
    before = map_area.height()
    assert log.height() < panel.fit_height(), "the default row is too short for the list"
    _front(window, dock)
    assert panel.height() >= panel.fit_height(), "the list and its buttons show without scrolling"
    _front(window, log)
    assert map_area.height() == before
    _front(window, dock)
    window.mode_combo.setCurrentText("Units")
    for _ in range(3):
        _settle()
    assert map_area.height() == before, "leaving Triggers mode sends the tab back too"
    window.mode_combo.setCurrentText("Triggers")
    _front(window, dock)
    window.resizeDocks([dock], [dock.height() + 40], Qt.Vertical)
    _settle()
    dragged = map_area.height()
    _front(window, log)
    assert map_area.height() == dragged, "a height the user set is kept"


def test_a_window_resize_while_grown_still_puts_the_row_back(window):
    dock, log = window.objectives_dock, window.log_dock
    before = window.height() - window.centralWidget().height()
    _front(window, dock)
    window.resize(window.width(), window.height() + 100)
    _settle()
    _front(window, log)
    assert window.height() - window.centralWidget().height() == before


def test_a_layout_saved_while_grown_comes_back_with_the_default_row(window):
    from PyQt5.QtCore import QByteArray

    before = window.height() - window.centralWidget().height()
    _front(window, window.objectives_dock)
    assert window.height() - window.centralWidget().height() > before
    window._close_docks()
    window.objectives_dock.show()
    window._apply_default_layout()
    _settle()
    assert window.restoreState(QByteArray(settings.get_window_state()), WINDOW_STATE_VERSION)
    for _ in range(3):
        _settle()
    _front(window, window.log_dock)
    assert window.height() - window.centralWidget().height() == before


def test_reset_layout_docks_a_floating_dock_back_behind_the_log(window):
    dock = window.objectives_dock
    dock.setFloating(True)
    _settle()
    assert dock.isFloating() and dock.isVisible()
    window.reset_layout_action.trigger()
    _settle()
    assert not dock.isFloating() and dock.isVisible()
    assert window.log_dock in window.tabifiedDockWidgets(dock)
    assert _front_tab(window, window.log_dock) == "Status Log"


def test_a_layout_saved_before_the_objectives_dock_is_refused(tmp_path):
    from PyQt5.QtCore import Qt

    window = _open(tmp_path)
    window.addDockWidget(Qt.RightDockWidgetArea, window.log_dock)
    _settle()
    state = bytes(window.saveState(WINDOW_STATE_VERSION - 1))
    conftest.close_window(window)
    settings.set_window_state(state)
    (tmp_path / "again").mkdir()
    window = _open(tmp_path / "again")
    try:
        assert window.dockWidgetArea(window.log_dock) == Qt.BottomDockWidgetArea
        assert window.log_dock in window.tabifiedDockWidgets(window.objectives_dock)
    finally:
        conftest.close_window(window)


def test_the_tab_behind_the_log_holds_across_a_relaunch(tmp_path):
    window = _open(tmp_path)
    conftest.close_window(window)
    assert settings.get_window_state() is not None
    (tmp_path / "again").mkdir()
    window = _open(tmp_path / "again")
    try:
        assert window.log_dock in window.tabifiedDockWidgets(window.objectives_dock)
        assert not window.objectives_dock.isHidden()
        assert _front_tab(window, window.log_dock) == "Status Log"
    finally:
        conftest.close_window(window)


def test_opening_the_type_picker_leaves_a_closed_dock_closed(window):
    dock = window.objectives_dock
    dock.toggleViewAction().trigger()
    _settle()
    assert dock.isHidden()
    window._on_trigger_detail_shown()
    _settle()
    assert dock.isHidden()


def test_a_load_outside_triggers_mode_clears_the_list_and_drops_the_old_document(window, tmp_path):
    panel = window.objectives_panel
    panel.toggle_tick(1)
    window.mode_combo.setCurrentText("View")
    _settle()
    (tmp_path / "again").mkdir()
    window.load_scenario(_objectives_file(tmp_path / "again"))
    _settle()
    assert panel.rows() == [] and panel.ticked() == set()
    assert window._objectives_doc is None
    window.mode_combo.setCurrentText("Triggers")
    _settle()
    assert _order(window) == [0, 1, 2]


def test_deleting_a_trigger_drops_the_ticks(window):
    panel = window.objectives_panel
    panel.toggle_tick(2)
    window.trigger_structural_edit("delete", [3])
    assert panel.ticked() == set()


def test_ticks_survive_a_field_edit_and_drop_when_renumber_ids_reorders_the_list(window):
    panel = window.objectives_panel
    panel.toggle_tick(1)
    window._objective_flags([1], "mute_objectives", True)
    window.trigger_structural_edit("move_down", [1])
    assert panel.ticked() == {1}, "a field edit or a display move leaves every trigger at its index"
    window.trigger_structural_edit("renumber", [])
    assert [t.short_description for t in _triggers(window)][:3] != [b[5] for b in _BAKED], "the list reordered"
    assert panel.ticked() == set(), "the tick would sit on another trigger"


def test_renumber_counts_the_list_down_from_the_top_value(window):
    panel = window.objectives_panel
    before = _records(window)
    panel.renumber_button.click()
    assert _records(window) == before + 1
    assert [row.order for row in panel.rows()] == [100, 99, 98]


def test_add_trigger_makes_the_promoted_row_current_in_both_panels(window, monkeypatch):
    panel = window.objectives_panel
    monkeypatch.setattr(panel, "pick_triggers", lambda candidates: [3])
    panel.select_index(0)
    panel.add_trigger_button.click()
    assert panel.current_index() == 3 and panel.selected_indices() == [3]
    assert window.trigger_panel.current_trigger_index() == 3
    panel.down_button.click()
    assert _order(window) == [0, 1, 3, 2], "the next move acts on the promoted row"


def test_a_row_click_keeps_a_trigger_filter_that_shows_the_trigger(window):
    trigger_panel = window.trigger_panel
    name = _triggers(window)[2].name
    trigger_panel.filter_edit.setText(name)
    _settle()
    window.objectives_panel.tree.setCurrentItem(_item(window, 2))
    _settle()
    assert trigger_panel.filter_edit.text() == name
    assert trigger_panel.current_trigger_index() == 2
    window.objectives_panel.tree.setCurrentItem(_item(window, 1))
    _settle()
    assert trigger_panel.filter_edit.text() == "", "a hidden trigger clears the filter"
    assert trigger_panel.current_trigger_index() == 1


def test_the_overlay_follows_a_window_resize(window):
    panel = window.objectives_panel
    panel.place_combo.setCurrentIndex(1)
    _settle()
    overlay = window.objectives_overlay
    before = overlay.width()
    window.resize(window.width() - 500, window.height())
    _settle()
    viewport = window.map_view.viewport().width()
    assert overlay.width() <= viewport // 2
    assert overlay.width() != before or before <= viewport // 2


def test_hiding_rows_leaves_them_out_of_the_preview_only(window):
    from descape.objectives_panel import COL_SHORT, HIDDEN_TIP

    panel = window.objectives_panel
    before = _records(window)
    panel.set_hidden([1], True)
    assert [entry.index for entry in panel.box.entries()] == [0, 2]
    assert [entry.index for entry in window.objectives_overlay.entries()] == [0, 2]
    assert _order(window) == [0, 1, 2], "the list keeps every row"
    assert _item(window, 1).toolTip(COL_SHORT) == HIDDEN_TIP
    assert _records(window) == before and not window.edit_history.is_dirty
    window._objective_flags([2], "mute_objectives", True)
    assert panel.hidden() == {1}, "an edit keeps the hidden rows"
    panel.set_hidden([1], False)
    assert [entry.index for entry in panel.box.entries()] == [0, 1, 2]


def test_hidden_rows_clear_on_another_document(window, tmp_path):
    panel = window.objectives_panel
    panel.set_hidden([1, 2], True)
    (tmp_path / "again").mkdir()
    window.load_scenario(_objectives_file(tmp_path / "again"))
    _settle()
    assert panel.hidden() == set()
    assert len(panel.box.entries()) == 3


def test_the_overlay_sits_in_the_maps_top_right_corner(window):
    from descape.objectives_box import OVERLAY_MARGIN_PX

    window.objectives_panel.place_combo.setCurrentIndex(1)
    _settle()
    overlay = window.objectives_overlay
    viewport = window.map_view.viewport().geometry()
    assert overlay.geometry().right() == viewport.right() - OVERLAY_MARGIN_PX
    assert overlay.geometry().left() >= viewport.center().x()
    assert overlay.geometry().top() == viewport.top() + OVERLAY_MARGIN_PX


def _many_objectives(window, count: int = 40) -> None:
    """`count` new objectives, so the box outgrows the dock and the map."""
    for _ in range(count):
        window._objective_new("objective", None)


def test_a_long_preview_scrolls_in_the_dock(window):
    panel = window.objectives_panel
    _many_objectives(window)
    _settle()
    bar = panel.box_scroll.verticalScrollBar()
    assert panel.box.height() > panel.box_scroll.viewport().height()
    assert bar.maximum() > 0
    assert bar.maximum() == panel.box.height() - panel.box_scroll.viewport().height(), "the last line is reachable"
    bar.setValue(bar.maximum())
    _settle()
    assert panel.box.y() == -bar.maximum()


def test_the_wheel_scrolls_a_long_map_box_and_zooms_elsewhere(window):
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    window.objectives_panel.place_combo.setCurrentIndex(1)
    _many_objectives(window)
    _settle()
    overlay = window.objectives_overlay
    viewport = window.map_view.viewport()
    assert overlay.overflow() > 0

    def wheel(pos: QPoint) -> None:
        event = QWheelEvent(QPointF(pos), QPointF(viewport.mapToGlobal(pos)), QPoint(), QPoint(0, -120),
                            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(viewport, event)
        _settle()

    centre = viewport.mapFromParent(overlay.geometry().center())
    wheel(centre)
    assert overlay.scroll_offset() > 0
    for _ in range(200):
        wheel(centre)
    assert overlay.scroll_offset() == int(overlay.overflow()), "stops at the end"
    (index, rect), *_ = overlay.check_rects()
    assert rect.top() < 0 or index not in (0, 1), "the check rects move with the scroll"
    zoom = window.map_view.transform().m11()
    scroll = overlay.scroll_offset()
    wheel(QPoint(20, viewport.height() - 20))
    assert overlay.scroll_offset() == scroll
    assert window.map_view.transform().m11() != zoom, "outside the box the wheel still zooms"


def test_a_plus_draws_a_square_checkbox_and_a_minus_a_round_one(window):
    from descape.objectives_box import layout
    from descape.objectives_model import OPTIONAL, REQUIRED

    entries = window.objectives_panel.box.entries()
    lines = {entry.index: (entry.marker, entry.text) for entry, *_ in layout(entries, 900)[0] if entry.index >= 0}
    assert lines == {
        0: (None, "Main Objective(s)"),
        1: (REQUIRED, "Destroy the Invaders"),
        2: (OPTIONAL, "<RED>Defend the Castle"),
    }
    assert [index for index, _rect in window.objectives_panel.box.check_rects()] == [1, 2], "a header has no checkbox"


def test_the_eye_column_hides_rows_from_the_preview_only(window):
    from PyQt5.QtCore import Qt

    from descape.objectives_panel import COL_PREVIEW

    panel = window.objectives_panel
    before = _records(window)
    panel.select_indices([1, 2])
    _item(window, 1).setCheckState(COL_PREVIEW, Qt.Unchecked)
    _settle()
    assert panel.hidden() == {1, 2}, "every selected row"
    assert [entry.index for entry in panel.box.entries()] == [0]
    assert _item(window, 1).checkState(COL_PREVIEW) == Qt.Unchecked
    assert _records(window) == before and not window.edit_history.is_dirty
    _item(window, 2).setCheckState(COL_PREVIEW, Qt.Checked)
    _settle()
    assert panel.hidden() == set()


def test_a_read_only_file_keeps_the_eye_column_but_not_the_flags(window):
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest

    from descape.objectives_panel import COL_MUTE, COL_PREVIEW

    window.scenario.trigger_write_supported = False
    window.refresh_objectives()
    panel = window.objectives_panel
    tree = panel.tree
    window.objectives_dock.raise_()
    _settle()

    def click(column: int) -> None:
        rect = tree.visualItemRect(_item(window, 1))
        x = tree.header().sectionViewportPosition(column) + 8
        QTest.mouseClick(tree.viewport(), Qt.LeftButton, pos=QPoint(x, rect.center().y()))
        _settle()

    click(COL_MUTE)
    assert _item(window, 1).checkState(COL_MUTE) == Qt.Unchecked
    assert not _triggers(window)[1].mute_objectives
    click(COL_PREVIEW)
    assert panel.hidden() == {1}


def _variable_text(window) -> str:
    """The cart objective's line as the box paints it, its variable tags resolved."""
    from descape import message_markup
    from descape.objectives_box import layout, variable_text

    state = window.objectives_panel.box.state
    entry = next(entry for entry, *_ in layout(state.entries, 900)[0] if entry.index == 1)
    return "".join(
        variable_text(run.text, state.variable_mode) if run.kind == message_markup.VARIABLE else run.text
        for run in message_markup.parse(entry.text, state.variables)
    )


def test_a_variable_tag_shows_its_name_value_or_both_and_persists(window):
    from descape.objectives_box import VARIABLE_BOTH, VARIABLE_NAME, VARIABLE_VALUE

    window.variable_structural_edit("add", -1, "TradeCarts")
    window._objective_short(1, "+ <tradecarts>/6 carts delivered.")
    panel = window.objectives_panel
    assert "TradeCarts" in panel.box.state.variables
    assert panel.variable_mode() == VARIABLE_NAME
    assert _variable_text(window) == "<tradecarts>/6 carts delivered.", "as typed, as the game draws it"
    panel.variable_combo.setCurrentIndex(1)
    assert settings.get_objectives_variables() == VARIABLE_VALUE
    assert _variable_text(window) == "0/6 carts delivered."
    assert window.objectives_overlay.state.variable_mode == VARIABLE_VALUE
    panel.variable_combo.setCurrentIndex(2)
    assert settings.get_objectives_variables() == VARIABLE_BOTH
    assert _variable_text(window) == "0 (tradecarts)/6 carts delivered."


def test_a_tag_naming_no_variable_stays_literal(window):
    window._objective_short(1, "+ <Nobody>/6 carts")
    window.objectives_panel.variable_combo.setCurrentIndex(1)
    assert _variable_text(window) == "<Nobody>/6 carts"


def test_columns_reorder_and_their_order_and_widths_survive_a_relaunch(tmp_path):
    from descape.objectives_panel import COL_SHORT, COL_TRIGGER

    window = _open(tmp_path)
    header = window.objectives_panel.tree.header()
    assert header.sectionsMovable()
    assert header.stretchLastSection() and header.logicalIndex(header.count() - 1) == COL_TRIGGER
    from descape.objectives_panel import COL_MUTE

    header.moveSection(header.visualIndex(COL_MUTE), 0)
    header.resizeSection(COL_SHORT, 333)
    window.objectives_panel._columns_timer.timeout.emit()
    assert settings.get_objectives_columns()
    conftest.close_window(window)
    (tmp_path / "again").mkdir()
    window = _open(tmp_path / "again")
    try:
        header = window.objectives_panel.tree.header()
        assert header.visualIndex(COL_MUTE) == 0
        assert header.logicalIndex(header.count() - 1) == COL_TRIGGER, "Trigger still fills the rest"
        assert header.sectionSize(COL_SHORT) == 333
    finally:
        conftest.close_window(window)


def test_an_unreadable_column_state_keeps_the_default(window):
    from descape.objectives_panel import COL_TRIGGER

    window.objectives_panel.restore_columns(b"not a header state")
    header = window.objectives_panel.tree.header()
    assert header.count() == COL_TRIGGER + 1
    assert header.logicalIndex(header.count() - 1) == COL_TRIGGER


def test_select_all_and_the_clipboard_keys_act_on_the_focused_objectives_list(window):
    panel = window.objectives_panel
    window.objectives_dock.raise_()
    _settle()
    window.trigger_panel.select_triggers([3])
    panel.tree.setFocus()
    _settle()
    assert window._objectives_list_focused()
    window._dispatch_select_all()
    assert panel.selected_indices() == [0, 1, 2]
    assert window.trigger_panel.selected_trigger_indices() != [0, 1, 2, 3], "the trigger list keeps its selection"
    count = len(_triggers(window))
    before = _records(window)
    window._dispatch_cut()
    window._dispatch_paste()
    assert len(_triggers(window)) == count and _records(window) == before, "no trigger is cut or pasted"
    window._dispatch_deselect()
    assert panel.selected_indices() == []


def test_typing_in_trigger_details_shows_in_the_preview_before_it_commits(window):
    from PyQt5.QtTest import QTest
    from trigger_panel_helpers import _row_widget

    from descape.objectives_panel import COL_SHORT

    panel = window.objectives_panel
    panel.tree.setCurrentItem(_item(window, 1))
    _settle()
    widget = _row_widget(window.trigger_panel, "short_description")
    widget.setFocus()
    widget.selectAll()
    before = _records(window)
    QTest.keyClicks(widget, "+ Live text")
    assert [entry.text for entry in panel.box.entries()][1] == "+ Live text"
    assert _item(window, 1).text(COL_SHORT) == "+ Live text"
    assert _triggers(window)[1].short_description == "+ Destroy the Invaders", "nothing written while typing"
    assert _records(window) == before
    panel.tree.setFocus()
    _settle()
    assert _triggers(window)[1].short_description == "+ Live text"
    assert _records(window) == before + 1


def _string_id_row(window, stid: int = 44737) -> None:
    window._objective_write("fixture", {1: {"short_description_stid": stid}})


@pytest.mark.parametrize(("answer", "text", "stid"), [("Yes", "+ Mine", -1), ("No", "+ Mine", 44737), ("Cancel", None, 44737)])
def test_an_inline_edit_on_a_string_id_row_asks_to_clear_the_id(window, monkeypatch, answer, text, stid):
    from PyQt5.QtWidgets import QMessageBox

    _string_id_row(window)
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *args: asked.append(args[2]) or getattr(QMessageBox, answer))
    )
    before = _records(window)
    window._objective_short(1, "+ Mine")
    assert asked and "#44737" in asked[0]
    trigger = _triggers(window)[1]
    assert trigger.short_description == (text or "+ Destroy the Invaders")
    assert trigger.short_description_stid == stid
    assert _records(window) == before + (0 if answer == "Cancel" else 1), "text and id are one step"


def test_use_my_text_clears_the_string_ids_of_the_selected_rows(window):
    _string_id_row(window)
    before = _records(window)
    window._objective_clear_string_ids([1])
    assert _triggers(window)[1].short_description_stid == -1
    assert _records(window) == before + 1


def test_renumber_with_rows_selected_renumbers_only_those(window):
    panel = window.objectives_panel
    panel.select_indices([1, 2])
    panel.renumber_button.click()
    assert [row.order for row in panel.rows()] == [100, 99, 98]
    assert _triggers(window)[0].description_order == 100


def test_a_window_resize_saves_no_column_layout(window):
    window.objectives_dock.raise_()
    _settle()
    window.resize(window.width() - 200, window.height())
    _settle()
    window.objectives_panel._columns_timer.timeout.emit()
    assert settings.get_objectives_columns() is None, "only a user's drag saves"


def test_a_box_that_fits_never_overflows_at_a_fractional_scale(window, monkeypatch):
    window.objectives_panel.place_combo.setCurrentIndex(1)
    _settle()
    overlay = window.objectives_overlay
    for ratio in (1.25, 1.5, 1.75):
        monkeypatch.setattr(overlay, "devicePixelRatioF", lambda r=ratio: r)
        overlay.set_shown(True)
        assert overlay.overflow() == 0, ratio


def test_the_wheel_over_a_map_box_checkbox_scrolls_or_zooms(window):
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    window.objectives_panel.place_combo.setCurrentIndex(1)
    _settle()
    overlay = window.objectives_overlay
    hits = overlay._hits

    def wheel_on_first_box() -> None:
        (_index, rect), *_ = overlay.check_rects()
        pos = QPoint(round(rect.center().x()), round(rect.center().y()))
        event = QWheelEvent(QPointF(pos), QPointF(hits.mapToGlobal(pos)), QPoint(), QPoint(0, -120),
                            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(hits, event)
        _settle()

    zoom = window.map_view.transform().m11()
    wheel_on_first_box()
    assert window.map_view.transform().m11() != zoom, "a box that fits passes the wheel to the map"
    _many_objectives(window)
    _settle()
    assert overlay.overflow() > 0
    zoom = window.map_view.transform().m11()
    wheel_on_first_box()
    assert overlay.scroll_offset() > 0
    assert window.map_view.transform().m11() == zoom


def test_a_read_only_flag_cell_click_off_the_box_still_selects_the_row(window):
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest

    from descape.objectives_panel import COL_OBJECTIVE

    window.scenario.trigger_write_supported = False
    window.refresh_objectives()
    window.objectives_dock.raise_()
    _settle()
    tree = window.objectives_panel.tree
    header = tree.header()
    # Row 1: the test window's dock shows only the first two rows.
    rect = tree.visualItemRect(_item(window, 1))
    assert tree.viewport().rect().contains(rect.center())
    x = header.sectionViewportPosition(COL_OBJECTIVE) + header.sectionSize(COL_OBJECTIVE) - 3
    QTest.mouseClick(tree.viewport(), Qt.LeftButton, pos=QPoint(x, rect.center().y()))
    _settle()
    assert window.objectives_panel.selected_indices() == [1]
    assert _item(window, 1).checkState(COL_OBJECTIVE) == Qt.Checked


def test_the_edit_menu_works_in_an_inline_editor_and_is_off_on_the_list(window):
    from descape.objectives_panel import COL_SHORT

    panel = window.objectives_panel
    window.objectives_dock.raise_()
    panel.tree.setFocus()
    _settle()
    window._update_clipboard_actions()
    assert not window.cut_action.isEnabled() and not window.paste_action.isEnabled()
    tree = panel.tree
    item = _item(window, 1)
    tree.editItem(item, COL_SHORT)
    editor = tree.indexWidget(tree.indexFromItem(item, COL_SHORT))
    editor.setFocus()
    _settle()
    window._dispatch_select_all()
    assert editor.selectedText() == editor.text(), "Select All selects the editor's text"
    window._dispatch_cut()
    assert editor.text() == "", "Cut reaches the editor"


def test_a_header_starting_with_a_sign_gets_no_checkbox(window):
    from descape.objectives_box import layout

    window._objective_short(0, "+ Main Objective(s)")
    lines = {entry.index: entry.marker for entry, *_ in layout(window.objectives_panel.box.entries(), 900)[0]}
    assert lines[0] is None


def test_box_text_is_drawn_at_its_on_screen_size(window, monkeypatch):
    from PyQt5.QtGui import QPainter, QPixmap

    from descape import objectives_box
    from descape.objectives_box import BODY_PX, DOCK_SCALE, layout, paint_box

    drawn = []
    monkeypatch.setattr(
        objectives_box,
        "_paint_runs",
        lambda painter, entry, rect, font, *rest: drawn.append((entry.kind, font.pixelSize(), painter.transform().m11())),
    )
    box = window.objectives_panel.box
    pixmap = QPixmap(400, 400)
    painter = QPainter(pixmap)
    painter.scale(DOCK_SCALE, DOCK_SCALE)
    paint_box(painter, layout(box.entries(), 900), 900, box.state)
    painter.end()
    body = [(px, scale) for kind, px, scale in drawn if kind not in ("Objectives", "header")]
    assert body and all(px == round(BODY_PX * DOCK_SCALE) and scale == 1 for px, scale in body), (
        "a canvas-size font shrunk by the painter smears"
    )


def test_the_preview_goes_below_the_list_in_a_tall_panel(window):
    from PyQt5.QtCore import Qt

    panel = window.objectives_panel
    panel.resize(500, 900)
    assert panel.splitter.orientation() == Qt.Vertical
    panel.resize(1400, 400)
    assert panel.splitter.orientation() == Qt.Horizontal


def test_flag_columns_stay_checkbox_narrow_with_headers_that_fit(window):
    from descape.objectives_panel import FLAG_COLUMNS

    tree = window.objectives_panel.tree
    metrics = tree.header().fontMetrics()
    labels = [tree.headerItem().text(column) for column, *_ in FLAG_COLUMNS]
    assert labels == ["H", "Obj", "Scr", "Mute"]
    widths = {tree.columnWidth(column) for column, *_ in FLAG_COLUMNS}
    assert len(widths) == 1
    for column, label, *_ in FLAG_COLUMNS:
        assert metrics.horizontalAdvance(label) + 8 <= tree.columnWidth(column), f"{label} is cut off"
        assert tree.headerItem().toolTip(column), f"{label} has no tooltip naming its flag"


def test_a_narrow_panel_scrolls_only_the_preview_across(window):
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QScrollArea

    from descape.objectives_panel import TICK_NOTE, ObjectivesPanel

    panel = ObjectivesPanel()
    panel.show_rows(window.objectives_panel.rows(), True)
    panel.resize(300, 800)
    panel.show()
    try:
        _settle()
        outer = panel.findChild(QScrollArea, options=Qt.FindDirectChildrenOnly)
        assert outer.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert outer.widget().width() <= outer.viewport().width(), "the list and its controls never scroll sideways"
        assert panel.box_scroll.horizontalScrollBar().maximum() > 0, "the preview keeps the game box's width and scrolls"
        assert panel.box.toolTip() == TICK_NOTE
    finally:
        panel.close()
        panel.deleteLater()
