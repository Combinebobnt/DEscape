"""Trigger unit references through a real offscreen ViewerWindow: the panel's
resolved labels for Unit/Unit[] fields, and Pick from map (arm, click, write,
disarm), against tests/fixtures/triggers_120x120.aoe2scenario."""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
PLAYER_ONE = 1


def _gen():
    return conftest.load_verify_module("gen_trigger_fixture")


def _window(style: str = "Flat"):
    window = conftest.shown_window(1100, 760)
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_trigger(_gen().REFERENCE_TRIGGER)
    return window


def _row(panel, field):
    for spec, _kind, _index, widget in panel._rows:
        if spec.name == field:
            return widget
    raise AssertionError(f"no {field} row in {[s.name for s, *_ in panel._rows]}")


def _texts(widget):
    from PyQt5.QtWidgets import QLabel

    return [label.text() for label in widget.findChildren(QLabel)]


def _describe(window, ref_id):
    from descape import unit_references

    return unit_references.describe(unit_references.build_reference_index(window.scenario), ref_id)


def _live(window, kind, index, field):
    gen = _gen()
    trigger = window.trigger_edits.manager().triggers[gen.REFERENCE_TRIGGER] if window.trigger_edits else None
    if trigger is None:
        from descape.scenario_io import parse_triggers

        trigger = parse_triggers(window.scenario).triggers[gen.REFERENCE_TRIGGER]
    entries = trigger.conditions if kind == "condition" else trigger.effects
    return getattr(entries[index], field)


def _records(window) -> int:
    return len(window.edit_history.records)


def _target(window, kind, index, field, is_list):
    return (_gen().REFERENCE_TRIGGER, kind, index, field, is_list)


# -- labels -------------------------------------------------------------------


def test_a_scalar_reference_shows_the_unit_it_names_and_follows_the_spinbox() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("condition", gen.DESTROY_CONDITION)
        host = _row(panel, "unit_object")
        house = _describe(window, gen.REF_HOUSE)
        assert house.endswith(f"[P1, X{gen.HOUSE_TILE[0]}, Y{gen.HOUSE_TILE[1]}]")
        assert house in _texts(host)
        panel.select_entry("condition", gen.DANGLING_CONDITION)
        assert f"{gen.DANGLING_REFERENCE_ID}: not placed on the map" in _texts(_row(panel, "unit_object"))
    finally:
        conftest.close_window(window)


def test_a_list_reference_shows_one_line_per_id() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", gen.TASK_OBJECT_EFFECT)
        expected = "\n".join(_describe(window, i) for i in gen.TASK_SELECTED_IDS)
        assert expected in _texts(_row(panel, "selected_object_ids"))
        assert _describe(window, gen.TASK_TARGET_ID) in _texts(_row(panel, "location_object_reference"))
    finally:
        conftest.close_window(window)


def test_a_long_list_is_capped_at_eight_lines() -> None:
    from descape.trigger_panel import TriggerPanel

    conftest.ensure_qapp()
    panel = TriggerPanel()
    panel.describe_unit_reference = lambda ref_id: f"unit {ref_id}"
    text = panel._unit_list_text(tuple(range(11)))
    assert text.splitlines() == [*(f"unit {i}" for i in range(8)), "... and 3 more"]
    assert panel._unit_list_text(tuple(range(8))).count("\n") == 7


def test_a_read_only_unit_field_names_a_dangling_id_once() -> None:
    """describe() already leads a dangling id with "999: ", so the read-only
    label must not prefix it again; a resolved unit still gets the id."""
    from descape import unit_references
    from descape.trigger_fields import REFERENCE, FieldSpec
    from descape.trigger_panel import TriggerPanel

    conftest.ensure_qapp()
    panel = TriggerPanel()
    spec = FieldSpec("unit_object", REFERENCE, presentation="Unit", read_only=True)
    panel.describe_unit_reference = lambda ref_id: unit_references.describe(unit_references.EMPTY_INDEX, ref_id)
    assert panel._display_text(spec, 999) == "999: not placed on the map"
    panel.describe_unit_reference = lambda ref_id: "Archer (4) [P1, X1, Y2]"
    assert panel._display_text(spec, 999) == "999: Archer (4) [P1, X1, Y2]"
    # An id that merely starts with the same digits is not a match.
    panel.describe_unit_reference = lambda ref_id: "9990: not placed on the map"
    assert panel._display_text(spec, 999) == "999: 9990: not placed on the map"


def test_moving_a_referenced_unit_updates_its_label() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", gen.TASK_OBJECT_EFFECT)
        before = _describe(window, gen.REF_ARCHER_A)
        unit = next(u for u in window.scenario.unit_manager.units[PLAYER_ONE] if u.reference_id == gen.REF_ARCHER_A)
        unit.x, unit.y = 70.5, 80.5
        window._after_unit_mutation()
        after = _describe(window, gen.REF_ARCHER_A)
        assert after != before and after.endswith("[P1, X70, Y80]")
        assert after in "\n".join(_texts(_row(panel, "selected_object_ids")))
    finally:
        conftest.close_window(window)


# -- Pick from map ----------------------------------------------------------------


def _key(ref_id):
    return (PLAYER_ONE, ref_id)


def test_arming_builds_the_pick_index_and_highlights_without_touching_the_selection() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", gen.TASK_OBJECT_EFFECT)
        assert window.map_view._unit_index is None, "Triggers mode builds no pick index by itself"
        window._selection = []
        window.arm_unit_picker(_target(window, "effect", gen.TASK_OBJECT_EFFECT, "selected_object_ids", True))
        assert window.map_view._unit_index is not None
        assert window.map_view.unit_picker_active()
        assert sorted(window.map_view._unit_select_keys) == sorted(_key(i) for i in gen.TASK_SELECTED_IDS)
        assert window._selection == []
        assert "Esc" in window.status_log.toPlainText().splitlines()[-1]
        window.disarm_unit_picker()
        assert not window.map_view.unit_picker_active()
        assert window.map_view._unit_index is None and window.map_view._unit_select_keys == []
    finally:
        conftest.close_window(window)


def test_a_scalar_pick_writes_once_and_disarms() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("condition", gen.DESTROY_CONDITION)
        records = _records(window)
        window.arm_unit_picker(_target(window, "condition", gen.DESTROY_CONDITION, "unit_object", False))
        window.map_view.on_unit_pick(_key(gen.REF_ARCHER_B))
        assert _live(window, "condition", gen.DESTROY_CONDITION, "unit_object") == gen.REF_ARCHER_B
        assert _records(window) == records + 1
        assert window._unit_picker is None and not window.map_view.unit_picker_active()
        assert _describe(window, gen.REF_ARCHER_B) in _texts(_row(panel, "unit_object"))
        assert window._selection == []
        window.undo()
        assert _live(window, "condition", gen.DESTROY_CONDITION, "unit_object") == gen.TASK_TARGET_ID
    finally:
        conftest.close_window(window)


def test_a_list_pick_toggles_in_and_out_and_stays_armed() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        effect = gen.PATROL_IDS_EFFECT
        panel.select_entry("effect", effect)
        records = _records(window)
        window.arm_unit_picker(_target(window, "effect", effect, "selected_object_ids", True))
        window.map_view.on_unit_pick(_key(gen.REF_HOUSE))
        assert list(_live(window, "effect", effect, "selected_object_ids")) == [*gen.PATROL_SELECTED_IDS, gen.REF_HOUSE]
        assert window._unit_picker is not None and window.map_view.unit_picker_active()
        assert (PLAYER_ONE, gen.REF_HOUSE) in window.map_view._unit_select_keys
        window.map_view.on_unit_pick(_key(gen.PATROL_SELECTED_IDS[0]))
        assert list(_live(window, "effect", effect, "selected_object_ids")) == [gen.REF_HOUSE]
        assert _records(window) == records + 2, "one undo record per click"
        assert window._unit_picker is not None
        assert window._selection == []
        window.undo()
        assert list(_live(window, "effect", effect, "selected_object_ids")) == [*gen.PATROL_SELECTED_IDS, gen.REF_HOUSE]
        assert window._unit_picker is None, "undo repopulates the panel, and disarms"
    finally:
        conftest.close_window(window)


def test_a_synthesized_click_on_a_unit_picks_it() -> None:
    from PyQt5.QtCore import QEvent, Qt

    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("condition", gen.DESTROY_CONDITION)
        window.arm_unit_picker(_target(window, "condition", gen.DESTROY_CONDITION, "unit_object", False))
        view = window.map_view
        tile = tuple(int(c) for c in gen.ARCHER_A_POS)
        view.centerOn(view._tile_polygon(*tile).boundingRect().center())
        pos = conftest.polygon_viewport_pos(view, *tile)
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
        assert _live(window, "condition", gen.DESTROY_CONDITION, "unit_object") == gen.REF_ARCHER_A
        assert not view.unit_picker_active()
        assert window._selection == []
    finally:
        conftest.close_window(window)


def _escape(view):
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))


@pytest.mark.parametrize("how", ["escape", "right_click", "button", "mode", "entry", "trigger", "tool"])
def test_each_cancel_disarms_without_writing(how) -> None:
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent

    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        effect = gen.TASK_OBJECT_EFFECT
        panel.select_entry("effect", effect)
        records = _records(window)
        target = _target(window, "effect", effect, "selected_object_ids", True)
        panel._pick_buttons[target].click()
        assert window._unit_picker == target
        assert panel._pick_buttons[target].isChecked()
        view = window.map_view
        if how == "escape":
            _escape(view)
        elif how == "right_click":
            view.mousePressEvent(
                QMouseEvent(QEvent.MouseButtonPress, QPointF(5, 5), Qt.RightButton, Qt.RightButton, Qt.NoModifier)
            )
        elif how == "button":
            panel._pick_buttons[target].click()
        elif how == "mode":
            window.mode_combo.setCurrentText("View")
        elif how == "entry":
            panel.select_entry("effect", gen.PATROL_IDS_EFFECT)
        elif how == "tool":
            # Triggers mode's only other tool (TASK-088 step 5).
            window.ruler_action.trigger()
            assert window._current_tool == "ruler"
        else:
            panel.select_trigger(0)
        assert window._unit_picker is None and not view.unit_picker_active()
        assert _records(window) == records
        assert list(_live(window, "effect", effect, "selected_object_ids")) == list(gen.TASK_SELECTED_IDS)
    finally:
        conftest.close_window(window)


def test_a_stale_target_does_not_write() -> None:
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("condition", gen.DESTROY_CONDITION)
        records = _records(window)
        stale = _target(window, "condition", gen.DANGLING_CONDITION, "unit_object", False)
        assert panel.apply_picked_reference(stale, gen.REF_ARCHER_A) is False
        assert _records(window) == records
        assert _live(window, "condition", gen.DANGLING_CONDITION, "unit_object") == gen.DANGLING_REFERENCE_ID
    finally:
        conftest.close_window(window)


def test_repeated_picks_on_a_stack_walk_down_it() -> None:
    from descape import unit_pick

    first, second, third = (f"u{i}" for i in range(3))
    members = [first, second, third]

    class _Entry:
        def __init__(self, name):
            self.player_id, self.unit = 1, type("U", (), {"reference_id": name})()

    entries = [_Entry(m) for m in members]
    chosen, index = unit_pick.stack_cycle_step(entries, entries[0], None)
    assert (chosen, index) == (entries[0], 0)
    chosen, index = unit_pick.stack_cycle_step(entries, entries[0], index)
    assert (chosen, index) == (entries[1], 1)
    chosen, index = unit_pick.stack_cycle_step(entries, entries[0], 2)
    assert (chosen, index) == (entries[0], 0)
    assert unit_pick.stack_cycle_step([], entries[2], None) == (entries[2], -1)


def test_a_none_defaulted_unit_list_is_a_list_editor_not_a_spinbox() -> None:
    """task_object and 8 other effects default selected_object_ids to None, not
    [], so the field is routed by presentation: a spinbox there showed
    "(unset)" for a stored list and would have written an int into it."""
    from PyQt5.QtWidgets import QAbstractSpinBox, QLineEdit

    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", gen.TASK_OBJECT_EFFECT)
        host = _row(panel, "selected_object_ids")
        assert not host.findChildren(QAbstractSpinBox)
        (editor,) = host.findChildren(QLineEdit)
        assert editor.text() == ", ".join(str(i) for i in gen.TASK_SELECTED_IDS)
        editor.setText(str(gen.REF_ARCHER_B))
        editor.editingFinished.emit()
        assert list(_live(window, "effect", gen.TASK_OBJECT_EFFECT, "selected_object_ids")) == [gen.REF_ARCHER_B]
        assert _texts(host) == [_describe(window, gen.REF_ARCHER_B)]
    finally:
        conftest.close_window(window)


def test_every_none_defaulted_unit_list_builds_a_list_editor_in_every_version() -> None:
    """TASK-088 step 7, through the panel's own widget builder. Pass conditions, for each
    effect of each shipped vocabulary whose selected_object_ids defaults to None (nine names
    in all, change_object_caption among them): _build_widget() for a stored [500, 501] gives
    no spinbox, one line edit reading "500, 501", and the resolved line per id under it."""
    from PyQt5.QtWidgets import QAbstractSpinBox, QLineEdit

    from descape import library_compat, trigger_fields, unit_references
    from descape.trigger_panel import TriggerPanel

    conftest.ensure_qapp()
    panel = TriggerPanel()
    panel.describe_unit_reference = lambda ref_id: f"unit {ref_id}"
    built = set()
    for version in library_compat.vocabulary_versions():
        vocabulary = library_compat.load_vocabulary(version)
        for definition in vocabulary.effects.values():
            for spec in trigger_fields.field_specs(definition, vocabulary.effect_presentation, "effect_type"):
                if spec.presentation != unit_references.LIST_PRESENTATION:
                    continue
                if definition.default_attributes.get(spec.name) is not None:
                    continue
                where = (version, definition.name)
                host = panel._build_widget(spec, [("effect", 0)], [500, 501])
                assert not host.findChildren(QAbstractSpinBox), where
                (editor,) = host.findChildren(QLineEdit)
                assert editor.text() == "500, 501", where
                assert "unit 500\nunit 501" in _texts(host), where
                built.add(definition.name)
    assert "change_object_caption" in built and len(built) == 9, sorted(built)


# -- the owed in-app passes, offscreen (TASK-088) ------------------------------------


def test_typing_an_unplaced_id_into_the_scalar_field_relabels_it_not_placed() -> None:
    """TASK-088 step 2. Pass conditions, the Destroy Object condition selected (its unit_object
    names the house): typing 777, an id no unit carries, into the field's spinbox and pressing
    Enter writes 777, and the line under the spinbox reads exactly "777: not placed on the map"
    in place of the house's line."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QSpinBox

    gen = _gen()
    unplaced = 777
    assert unplaced not in gen.PLACED_REFERENCE_IDS
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("condition", gen.DESTROY_CONDITION)
        host = _row(panel, "unit_object")
        house = _describe(window, gen.REF_HOUSE)
        assert house in _texts(host)
        (spin,) = host.findChildren(QSpinBox)
        spin.setFocus()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, str(unplaced))
        QTest.keyClick(spin, Qt.Key_Return)
        assert _live(window, "condition", gen.DESTROY_CONDITION, "unit_object") == unplaced
        texts = _texts(_row(panel, "unit_object"))
        assert f"{unplaced}: not placed on the map" in texts
        assert house not in texts
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("garrisoned_shown", [False, True], ids=["default_filters", "nothing_filtered"])
def test_the_scalar_pick_button_forces_pan_says_so_and_unchecks_after_its_pick(garrisoned_shown) -> None:
    """TASK-088 step 3. Pass conditions, the Destroy Object condition selected and the Ruler
    the current tool: clicking the unit_object row's Pick button arms the picker on that field
    and checks the button, the tool is Pan (action checked, view tool too), and the status log's
    last line is exactly "Click a unit to set unit object; Esc to finish", plus
    " (units hidden by Filters can't be picked)" only while a filter hides something (the
    Filters menu ships Show Garrisoned unchecked, so the default carries it). Picking archer B
    writes its id, disarms, and leaves that field's Pick button unchecked; one Undo restores
    the house's id."""
    from descape.unit_filter import UnitFilter

    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        panel.select_entry("condition", gen.DESTROY_CONDITION)
        hint = "Click a unit to set unit object; Esc to finish"
        if garrisoned_shown:
            window.show_garrisoned_action.setChecked(True)
            assert window._unit_filter == UnitFilter()
        else:
            assert window._unit_filter != UnitFilter(), "the default Filters menu hides garrisoned units"
            hint += " (units hidden by Filters can't be picked)"
        window.ruler_action.trigger()
        assert window._current_tool == "ruler"
        target = _target(window, "condition", gen.DESTROY_CONDITION, "unit_object", False)
        button = panel._pick_buttons[target]
        assert button.text() == "Pick"
        button.click()
        assert window._unit_picker == target and button.isChecked()
        assert window._current_tool == "pan" and window.pan_action.isChecked()
        assert window.map_view._tool == "pan"
        assert window.status_log.toPlainText().splitlines()[-1] == hint
        window.map_view.on_unit_pick(_key(gen.REF_ARCHER_B))
        assert _live(window, "condition", gen.DESTROY_CONDITION, "unit_object") == gen.REF_ARCHER_B
        assert window._unit_picker is None
        assert not panel._pick_buttons[target].isChecked(), "Pick stayed checked after its one-shot pick"
        window.undo()
        assert _live(window, "condition", gen.DESTROY_CONDITION, "unit_object") == gen.REF_HOUSE
    finally:
        conftest.close_window(window)


def test_each_list_pick_through_the_objects_set_names_the_unit_in_the_status_log() -> None:
    """TASK-088 step 4, through the Objects row's Set (GH #138 moved the list Pick there). Pass
    conditions, the Task Object effect selected: Set arms the list picker, highlighting the
    field's two archers; picking the house adds it, stays armed and the status log's last line
    is exactly "Picked " + the house's resolved line; picking archer A again removes it and the
    last line names archer A the same way; Esc then disarms and writes nothing more."""
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        effect = gen.TASK_OBJECT_EFFECT
        panel.select_entry("effect", effect)
        set_button = panel._group_rows["objects"].set_button
        set_button.click()
        target = _target(window, "effect", effect, "selected_object_ids", True)
        assert window._unit_picker == target
        assert sorted(window.map_view._unit_select_keys) == sorted(_key(i) for i in gen.TASK_SELECTED_IDS)

        def last_line() -> str:
            return window.status_log.toPlainText().splitlines()[-1]

        window.map_view.on_unit_pick(_key(gen.REF_HOUSE))
        assert window._unit_picker == target
        assert last_line() == f"Picked {_describe(window, gen.REF_HOUSE)}"
        window.map_view.on_unit_pick(_key(gen.REF_ARCHER_A))
        assert list(_live(window, "effect", effect, "selected_object_ids")) == [gen.REF_ARCHER_B, gen.REF_HOUSE]
        assert last_line() == f"Picked {_describe(window, gen.REF_ARCHER_A)}"
        records = _records(window)
        _escape(window.map_view)
        assert window._unit_picker is None and not set_button.isChecked()
        assert _records(window) == records
        assert list(_live(window, "effect", effect, "selected_object_ids")) == [gen.REF_ARCHER_B, gen.REF_HOUSE]
    finally:
        conftest.close_window(window)
