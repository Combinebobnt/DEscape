"""Trigger panel: entry rows in display order and entry moves (GH #213)."""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _ARMOUR_TRIGGER,
    TRIGGER_FIXTURE,
    _action,
    _ctrl_select,
    _entry_menu_move_states,
    _entry_move_states,
    _entry_rows,
    _group,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _ordered_window,
    _record_count,
    _row_widgets,
    _select_entries,
    _triggers_window,
    _types_of,
    _walk_every_row,
    _window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- GH #213: conditions and effects in display order ---------------------------
#
# The shipped fixture holds identity condition and effect orders, so every
# other trigger panel test is blind to a tree that lists entries in storage order.


def test_gh213_entry_rows_follow_the_display_order_and_carry_the_list_index(tmp_path: Path) -> None:
    from test_trigger_write_path import ENTRY_ORDER_TRIGGER, ENTRY_ORDERS

    window, _base = _ordered_window(tmp_path)
    try:
        panel = window.trigger_panel
        trigger = panel._manager().triggers[ENTRY_ORDER_TRIGGER]
        for kind, order in ENTRY_ORDERS.items():
            assert _entry_rows(panel, kind) == order
            group = _group(panel, kind)
            entries = getattr(trigger, f"{kind}s")
            for row, index in enumerate(order):
                item = group.child(row)
                assert (item.text(0), item.text(1)) == tuple(panel._describe(kind, entries[index])), (kind, row)
                assert panel._entry_item_for(kind, index) is item, (kind, row)
        assert panel._entry_item_for("effect", 7) is None

        # A trigger that never used the feature lists as it always did.
        panel.select_trigger(_ARMOUR_TRIGGER)
        assert _entry_rows(panel, "effect") == [0, 1]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh213_a_joiner_stored_last_is_listed_between_its_operands(tmp_path: Path) -> None:
    """The report's own case (examples/8tp3w9j, trigger 2): stored `A, B, OR`
    under order [0, 2, 1] is `A, OR, B` in the game editor, and was listed
    `A, B, OR` here."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel, set_entry_order

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = TriggerEditModel(loaded)
    trigger = model.manager().triggers[0]
    trigger._add_condition(5)  # objects in area
    trigger._add_condition(29)  # OR
    assert [c.condition_type for c in trigger.conditions] == [10, 5, 29], "fixture assumption: one timer to start"
    set_entry_order(trigger, "condition", [0, 2, 1])
    model.mark_dirty(0)
    base = tmp_path / "joiner.aoe2scenario"
    write_scenario(loaded, base, triggers=model)

    window = _window()
    try:
        window.load_scenario(base)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        panel.select_trigger(0)
        group = _group(panel, "condition")
        assert [group.child(row).text(0) for row in range(group.childCount())] == ["timer", "or", "objects in area"]
        assert _entry_rows(panel, "condition") == [0, 2, 1]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh213_selection_reads_in_display_order_and_selects_by_list_index(tmp_path: Path) -> None:
    from test_trigger_write_path import ENTRY_ORDER_TRIGGER

    window, _base = _ordered_window(tmp_path)
    try:
        panel = window.trigger_panel
        trigger = panel._manager().triggers[ENTRY_ORDER_TRIGGER]
        _select_entries(panel, ("effect", 0), ("effect", 4), ("condition", 1), ("condition", 2))
        # Conditions show as 2, 0, 3, 1 and effects as 4, 0, 6, 2, 5, 1, 3.
        assert panel.selected_entry_refs() == [("condition", 2), ("condition", 1), ("effect", 4), ("effect", 0)]
        assert panel.current_entry_ref() == ("effect", 0), "the first ref is the current row"
        assert panel.entry_tree.currentItem() is _group(panel, "effect").child(1)

        panel.select_entry("condition", 3)
        assert panel.entry_tree.currentItem() is _group(panel, "condition").child(2)
        assert panel._current_entry()[2] is trigger.conditions[3], "the form shows the row's own entry"

        # Conditions 2 and 3 are the two destroy_object ones: rows 0 and 2.
        panel.select_same_type_as(panel._entry_item_for("condition", 3))
        assert panel.selected_entry_refs() == [("condition", 2), ("condition", 3)]
        assert panel.current_entry_ref() == ("condition", 3)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh213_status_reasons_count_display_rows(tmp_path: Path) -> None:
    """"effect 3" is the third row of the Effects group, not list index 2, and
    the reasons read top to bottom as the rows do."""
    import re

    from test_trigger_write_path import ENTRY_ORDER_TRIGGER

    from descape.trigger_status import Status

    window, _base = _ordered_window(tmp_path)
    try:
        panel = window.trigger_panel
        rollup = panel.status_of(ENTRY_ORDER_TRIGGER).trigger.reasons
        # List indices: condition 3, effect 6 (patrol) and effect 3 (kill object).
        assert rollup == (
            "condition 3 (destroy object): unit 999 (unit object) is not placed on the map",
            "effect 3 (patrol): missing location",
            "effect 7 (kill object): area is only partly set",
        )
        for kind in ("condition", "effect"):
            group = _group(panel, kind)
            reasons = group.data(0, panel._STATUS_ROLE).reasons
            assert reasons == tuple(reason for reason in rollup if reason.startswith(kind))
            flagged = {int(re.match(rf"{kind} (\d+) ", reason).group(1)) - 1 for reason in reasons}
            for row in range(group.childCount()):
                status = group.child(row).data(0, panel._STATUS_ROLE).status
                assert (status is Status.PROBLEM) == (row in flagged), (kind, row)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh213_browsing_a_non_identity_file_changes_nothing_and_an_edit_keeps_its_orders(tmp_path: Path) -> None:
    from test_trigger_write_path import ENTRY_ORDER_TRIGGER, ENTRY_ORDERS

    from descape import trigger_fields
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window, base = _ordered_window(tmp_path)
    try:
        panel = window.trigger_panel
        _walk_every_row(panel)
        assert window.trigger_edits is None, "browsing must not build an edit model"
        assert not window.edit_history.is_dirty, "browsing must record no undo step"
        trigger = panel._manager().triggers[ENTRY_ORDER_TRIGGER]
        # The raw arrays, not the healing getters.
        assert trigger._condition_order == ENTRY_ORDERS["condition"]
        assert trigger._effect_order == ENTRY_ORDERS["effect"]
        browsed = tmp_path / "browsed.aoe2scenario"
        write_scenario(window.scenario, browsed, triggers=window.trigger_edits)
        assert browsed.read_bytes() == base.read_bytes()

        # One field edit re-serializes the trigger, and the orders ride along.
        panel.select_trigger(ENTRY_ORDER_TRIGGER)
        spec = trigger_fields.FieldSpec("timer", trigger_fields.INT)
        window.set_entry_field(ENTRY_ORDER_TRIGGER, "condition", 0, spec, 9)
        for kind, order in ENTRY_ORDERS.items():
            assert _entry_rows(panel, kind) == order
        edited = tmp_path / "edited.aoe2scenario"
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)

        window.undo()
        for kind, order in ENTRY_ORDERS.items():
            assert _entry_rows(panel, kind) == order, "an undo deep-copies the state back into the trigger, which must keep the order"
        undone = tmp_path / "undone.aoe2scenario"
        write_scenario(window.scenario, undone, triggers=window.trigger_edits)
        assert undone.read_bytes() == base.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()

    reloaded_scenario = load_map_and_units(edited)  # kept alive: the library's store is weak
    reloaded = parse_triggers(reloaded_scenario)
    assert reloaded is not None
    after = reloaded.triggers[ENTRY_ORDER_TRIGGER]
    assert after.conditions[0].timer == 9
    assert {kind: list(getattr(after, f"{kind}_order")) for kind in ENTRY_ORDERS} == ENTRY_ORDERS


# -- GH #213: Move Up / Move Down for conditions and effects --------------------


def test_gh213_the_entry_row_ends_in_a_named_glyph_pair() -> None:
    from PyQt5.QtWidgets import QToolButton

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        row = panel.entry_buttons
        assert [w.text() for w in _row_widgets(row)] == ["New", "Duplicate", "Delete", "Type…"]
        moves = row.itemAt(row.count() - 1).layout()
        assert _row_widgets(moves) == [panel.entry_move_up_button, panel.entry_move_down_button]
        assert moves.spacing() == 2
        for button, glyph, name in (
            (panel.entry_move_up_button, "▲", "Move Entry Up"),
            (panel.entry_move_down_button, "▼", "Move Entry Down"),
        ):
            assert isinstance(button, QToolButton)
            assert (button.text(), button.accessibleName()) == (glyph, name)
        # Apart from the trigger row's pair, which an AT-SPI locator finds by name.
        assert panel.trigger_move_up_button.accessibleName() == "Move Up"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh213_the_entry_button_row_needs_no_more_width_than_before() -> None:
    """Pre-registered: 343 px (the four-button row's minimum before ▲▼ joined
    it), 347 at most. Measured 340, offscreen Qt 5.15.19, DejaVu Sans 12pt at
    96 dpi: the four push buttons give up Fusion's 80 px floor for the room."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert panel.entry_buttons.minimumSize().width() <= 347
        for button in _row_widgets(panel.entry_buttons):
            text = button.fontMetrics().horizontalAdvance(button.text())
            assert text < button.minimumSizeHint().width() <= button.sizeHint().width(), button.text()

        # The minimum follows a live UI font change (Settings > Appearance), with no refit step.
        button = panel.entry_new_button
        before = button.minimumSizeHint().width()
        font = button.font()
        font.setPointSize(font.pointSize() * 2)
        button.setFont(font)
        assert button.minimumSizeHint().width() > max(before, button.fontMetrics().horizontalAdvance("New"))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh213_the_entry_move_buttons_and_menu_follow_the_selection(tmp_path: Path) -> None:
    from test_trigger_write_path import ENTRY_ORDER_TRIGGER

    window, _base = _ordered_window(tmp_path)
    try:
        panel = window.trigger_panel
        up, down = panel.entry_move_up_button, panel.entry_move_down_button

        # The trigger row is current: no entry to move.
        assert _entry_move_states(panel) == (False, False)
        assert up.toolTip() == panel._ENTRY_MOVE_UP_TIP + " (select a condition or effect)"
        panel.entry_tree.setCurrentItem(_group(panel, "effect"))
        assert _entry_move_states(panel) == (False, False)

        # Conditions show as 2, 0, 3, 1 and effects as 4, 0, 6, 2, 5, 1, 3.
        cases = [
            ([("condition", 2)], (False, True)),  # the first row
            ([("condition", 1)], (True, False)),  # the last row
            ([("condition", 0)], (True, True)),
            ([("condition", 2), ("condition", 3)], (True, True)),  # rows 1 and 3: the lower one closes up
            ([("condition", 2), ("condition", 0)], (False, True)),  # a block against the top
            ([("condition", 3), ("condition", 1)], (True, False)),  # a block against the bottom
            ([("condition", i) for i in range(4)], (False, False)),  # the whole list
            ([("condition", 2), ("effect", 3)], (True, True)),  # first of one list, last of the other
            ([("condition", 2), ("effect", 4)], (False, True)),  # first of both
        ]
        for refs, expected in cases:
            _select_entries(panel, *refs)
            assert _entry_move_states(panel) == expected, refs
            assert _entry_menu_move_states(panel, refs[0]) == expected, refs
        _select_entries(panel, ("condition", 2))
        assert up.toolTip() == panel._ENTRY_MOVE_UP_TIP + " (already first)"
        assert down.toolTip() == panel._ENTRY_MOVE_DOWN_TIP
        _select_entries(panel, ("condition", 1))
        assert down.toolTip() == panel._ENTRY_MOVE_DOWN_TIP + " (already last)"
        menu = panel.entry_context_menu(panel._entry_item_for("condition", 1))
        # The button's reason, without its modifier-click hints (TASK-356).
        assert _action(menu, "Move Down").toolTip() == panel._ENTRY_MOVE_DOWN_MENU_TIP + " (already last)"
        assert down.toolTip().startswith(panel._ENTRY_MOVE_DOWN_MENU_TIP + ". Shift+click")
        assert not [a.text() for a in menu.actions() if "click" in a.toolTip()], "a menu entry ignores modifiers"

        # The type picker takes the lower pane, and the entry verbs with it.
        _select_entries(panel, ("condition", 0))
        panel._request_entry_op("new")
        assert _entry_move_states(panel) == (False, False)
        assert "close the type picker" in up.toolTip()
        panel._close_picker()
        assert _entry_move_states(panel) == (True, True)

        # Several triggers selected: the entry tree shows none of them.
        _ctrl_select(panel, 2, ENTRY_ORDER_TRIGGER)
        assert _entry_move_states(panel) == (False, False)
        assert "select a single trigger" in up.toolTip()

        # A file that cannot be written back.
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        panel.select_trigger(ENTRY_ORDER_TRIGGER)
        _select_entries(panel, ("condition", 0))
        assert _entry_move_states(panel) == (False, False)
        assert _entry_menu_move_states(panel, ("condition", 0)) == (False, False)
        assert "cannot be edited" in up.toolTip()
        panel._request_entry_op("move_up")
        assert window.trigger_edits is None, "a gated panel must not reach the model"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh213_move_up_moves_the_row_keeps_it_selected_and_undoes(tmp_path: Path) -> None:
    from test_trigger_write_path import ENTRY_ORDER_TRIGGER

    window, _base = _ordered_window(tmp_path)
    try:
        panel = window.trigger_panel
        listed = _types_of(window, ENTRY_ORDER_TRIGGER, "condition")
        _select_entries(panel, ("condition", 0))  # the second row
        records = _record_count(window)

        panel.entry_move_up_button.click()

        assert _entry_rows(panel, "condition") == [0, 2, 3, 1]
        assert _types_of(window, ENTRY_ORDER_TRIGGER, "condition") == listed, "a move permutes the order, never the list"
        assert panel.selected_entry_refs() == [("condition", 0)]
        assert panel.current_entry_ref() == ("condition", 0)
        assert panel.entry_tree.currentItem() is _group(panel, "condition").child(0)
        assert window.windowTitle().startswith("*")
        assert _record_count(window) == records + 1
        assert window.edit_history.records[-1].label == "Move condition up"
        assert _entry_move_states(panel) == (False, True), "it is the first row now"

        window.undo()
        assert _entry_rows(panel, "condition") == [2, 0, 3, 1]
        assert not window.windowTitle().startswith("*")

        # The menu's verb, on a selection across both lists, is one record too.
        _select_entries(panel, ("effect", 0), ("effect", 6), ("condition", 3))
        _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Move Down").trigger()
        assert _entry_rows(panel, "effect") == [4, 2, 0, 6, 5, 1, 3]
        assert _entry_rows(panel, "condition") == [2, 0, 1, 3]
        assert panel.selected_entry_refs() == [("condition", 3), ("effect", 0), ("effect", 6)]
        assert window.edit_history.records[-1].label == "Move 3 entries down"
        window.undo()
        assert (_entry_rows(panel, "condition"), _entry_rows(panel, "effect")) == (
            [2, 0, 3, 1], [4, 0, 6, 2, 5, 1, 3],
        ), "one undo takes both lists back"
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh213_a_move_that_changes_nothing_records_nothing(tmp_path: Path) -> None:
    """commit_trigger_edit() pushes unconditionally, so the funnel must not
    open an edit for a row already at the end it is sent to."""
    from test_trigger_write_path import ENTRY_ORDER_TRIGGER

    from descape.scenario_write import write_scenario

    window, base = _ordered_window(tmp_path)
    try:
        records = _record_count(window)
        window.entry_move(ENTRY_ORDER_TRIGGER, [("condition", 2)], -1)  # the first row, up
        assert _record_count(window) == records
        window.entry_move(ENTRY_ORDER_TRIGGER, [("effect", 3)], 1)  # the last row, down
        window.entry_move(ENTRY_ORDER_TRIGGER, [("condition", 9)], 1)  # a stale ref
        window.entry_move(9, [("condition", 0)], 1)  # no such trigger
        assert _record_count(window) == records
        assert not window.edit_history.is_dirty
        assert not window.windowTitle().startswith("*")
        out = tmp_path / "unmoved.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == base.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh213_repeated_moves_keep_the_scroll_and_the_moved_row_in_view() -> None:
    """Each move rebuilds the entry tree, which leaves it scrolled to the top."""
    from PyQt5.QtWidgets import QApplication

    from descape import brush

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        panel.select_entry("effect", 2)  # the fixture's Create Object
        window.brush_size_spin.setValue(5)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(brush.BRUSH_SHAPE_SQUARE))
        window.stamp_create_objects(50, 50)
        QApplication.processEvents()
        bar = panel.entry_tree.verticalScrollBar()
        viewport = panel.entry_tree.viewport().height()
        panel.select_entry("effect", 20)
        QApplication.processEvents()
        start = bar.value()
        assert start > 0, "fixture assumption: the entries overflow the tree, and this row sits below the fold"

        def in_view() -> bool:
            rect = panel.entry_tree.visualItemRect(panel._entry_item_for("effect", 20))
            return rect.top() >= 0 and rect.bottom() <= viewport

        for click in range(5):
            panel.entry_move_down_button.click()
            QApplication.processEvents()
            assert bar.value() >= start, f"click {click}: the list jumped up to {bar.value()}"
            assert in_view(), f"click {click}"
        down = bar.value()
        for click in range(10):
            panel.entry_move_up_button.click()
            QApplication.processEvents()
            assert down - (click + 1) <= bar.value() <= down, f"click {click}: scrolled to {bar.value()}"
            assert in_view(), f"click {click}"
        assert panel.selected_entry_refs() == [("effect", 20)]
    finally:
        window.edit_history.mark_saved()
        window.close()
