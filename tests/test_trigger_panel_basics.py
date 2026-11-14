"""Coverage for Triggers mode, the trigger browser (phase 4a.3), and its
property editor (phase 4b.6): mode entry, browsing, the form's fields and
references.

Two guarantees are pinned here, and the second is the one that is easy to break
without noticing.

1. **Containment.** Entering Triggers mode gates every edit tool off.
2. **Browsing is not editing.** Qt fires valueChanged/currentIndexChanged on a
   programmatic populate exactly as on a user edit, and editingFinished on a
   focus-out with unchanged text. commit_trigger_edit() pushes unconditionally,
   so either one reaching the window would record phantom undo steps and dirty
   triggers, and the file would stop saving byte-identically for a user who only
   looked at it. test_browsing_every_row_saves_byte_identically is the test that
   would catch that, and it is worth more than any single field-editing test
   below.

The width assertions moved with 4b.6: the two panes scroll independently now,
so the browser is readable at 340 px, where the single-tree 4a.3 layout needed
560 and still elided rows to "s...".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    TRIGGER_FIXTURE,
    _appended_effects,
    _effect_type_id,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _row_widget,
    _select_armour_split_effect,
    _select_entries,
    _select_first_condition,
    _triggers_window,
    _walk_every_row,
    _window,
)

from testkit.qt_window import panel_width, set_panel_width

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def test_triggers_mode_swaps_the_left_panel_and_gates_edit_tools() -> None:
    window = _window()
    try:
        assert window.left_stack.currentIndex() == 0
        window.mode_combo.setCurrentText("Triggers")
        assert window.mode == "triggers"
        assert window.left_stack.currentIndex() == 1
        # Same containment View mode has: no edit tool is reachable, and the
        # active tool falls back to Pan rather than staying checked-but-dead.
        assert not window.draw_action.isEnabled()
        assert not window.elevation_action.isEnabled()
        assert window.pan_action.isChecked()

        window.mode_combo.setCurrentText("View")
        assert window.left_stack.currentIndex() == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_mode_triggers_is_rebindable_and_bound() -> None:
    from PyQt5.QtGui import QKeySequence

    window = _window()
    try:
        assert "mode_triggers" in window._keybind_actions
        assert window.mode_triggers_action.shortcut() == QKeySequence("Ctrl+T")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_entering_triggers_mode_widens_a_too_narrow_pane() -> None:
    from descape.trigger_panel import TriggerPanel

    window = _window()
    try:
        set_panel_width(window, 200)
        window.mode_combo.setCurrentText("Triggers")
        assert panel_width(window) >= TriggerPanel.MIN_USEFUL_WIDTH
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_widening_never_shrinks_a_wider_user_chosen_pane() -> None:
    """The splitter position persists, so a width the user dragged to must
    survive a mode switch -- _widen_left_column only ever grows."""
    window = _window()
    try:
        set_panel_width(window, 900)
        window.mode_combo.setCurrentText("Triggers")
        assert panel_width(window) == 900
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_panel_reports_a_zero_trigger_document() -> None:
    """The shipped blank template has no triggers. Worth its own test: it is
    the one document the default tier can exercise the populate path with."""
    window = _window()
    try:
        window.load_scenario(BLANK_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        assert "0 trigger" in window.trigger_panel.status.text()
        assert window.trigger_panel.tree.topLevelItemCount() == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_closing_a_map_clears_the_panel() -> None:
    window = _window()
    try:
        window.load_scenario(BLANK_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        window.edit_history.mark_saved()
        window.close_scenario()
        assert window.trigger_panel.tree.topLevelItemCount() == 0
        assert window.trigger_panel.status.text() == window.trigger_panel._NO_DOCUMENT
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- 4b.6: the property editor ----------------------------------------------


@pytest.mark.parametrize("sort_index", [0, 1], ids=["display order", "file order"])
def test_browsing_every_row_saves_byte_identically(tmp_path: Path, sort_index: int) -> None:
    """The guarantee all of 4b exists to protect. A populate that leaked one
    signal through would build a model, dirty a trigger, and re-serialize the
    section, and nothing else in the suite would notice.

    Parametrized over both sort modes: the reordering work's row-index
    mapping (current_trigger_index() reading Qt.UserRole, _refresh_labels()
    translating through _item_for_index) touches every read path browsing
    exercises, so this is the one test in the suite positioned to catch a
    mapping bug that only shows up under a non-default sort.
    """
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        window.trigger_panel.sort_combo.setCurrentIndex(sort_index)
        _walk_every_row(window.trigger_panel)

        assert window.trigger_edits is None, "browsing must not build an edit model"
        assert not window.edit_history.is_dirty, "browsing must record no undo step"
        assert not window.windowTitle().startswith("*")

        out = tmp_path / "browsed.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_focus_out_with_unchanged_text_records_nothing() -> None:
    """editingFinished fires on focus-out whether or not the text changed, so
    tabbing through the form would otherwise record one edit per field."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        name = _row_widget(panel, "name")

        name.editingFinished.emit()
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_populating_a_widget_that_cannot_hold_the_live_value_records_nothing() -> None:
    """The _populating guard's real job, and the case the equality check cannot
    cover on its own.

    A QSpinBox cannot hold None and a QComboBox cannot hold a value outside its
    enum, so populate necessarily sets a *different* value than the field holds
    and emits valueChanged with it. Comparing new against current would see a
    genuine difference and report a phantom edit. Every field in the fixture
    happens to read back as a plain int, which is why this has to arrange the
    mismatch rather than wait for one.
    """
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_first_condition(panel)
        current = panel._current_entry()
        assert current is not None and current[0] == "condition"
        entry = current[2]

        spec = next((s for s, *_ in panel._rows if s.kind == "int"), None)
        assert spec is not None, "a condition must expose at least one int field"
        # Not routed through the model on purpose: this is the pre-model state a
        # populate runs in, and the assertion is that populate stays silent.
        setattr(entry, spec.name, None)

        panel._populate_property_form()

        assert window.trigger_edits is None, "a populate must never build an edit model"
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_change_reported_during_a_populate_is_ignored() -> None:
    """Explicitly white-box, because the guard it pins has no reachable failure
    through today's API: every widget in _build_widget() is connected after its
    value is set, so a populate emits nothing at all.

    The test exists because that ordering is the *only* thing making the guard
    unreachable. Connecting before setting in some later edit would put one
    phantom edit per field back on the table, and the equality check would not
    stop it: a spinbox cannot hold None and a combo cannot hold an out-of-enum
    value, so those populates genuinely differ from the live value.
    """
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_first_condition(panel)
        spec = next(s for s, *_ in panel._rows if s.kind == "int")

        panel._populating = True
        try:
            panel._changed(spec, [("condition", 0)], 4242)
        finally:
            panel._populating = False

        assert window.trigger_edits is None, "a populate-time signal must not build a model"
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_editing_a_trigger_name_through_the_widget_records_one_edit() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        name = _row_widget(panel, "name")

        name.setText("Renamed in the panel")
        name.editingFinished.emit()

        assert window.trigger_edits is not None
        assert window.trigger_edits.dirty_indices() == [0]
        assert panel.tree.topLevelItem(0).text(panel._COL_NAME) == "Renamed in the panel", (
            "the list row updates in place rather than waiting for a rebuild"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_toggling_a_bool_field_records_one_edit() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        looping = _row_widget(panel, "looping")
        before = looping.isChecked()

        looping.setChecked(not before)
        assert window.trigger_edits is not None
        assert window.trigger_edits.dirty_indices() == [0]
        assert bool(window.trigger_edits.manager().triggers[0].looping) is (not before)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_trigger_header_fields_round_trip_through_the_form(tmp_path: Path) -> None:
    """GH #141: both string table ids, description order and execute on load
    are form rows, and each edit reaches the saved file. "Fixture: setup" is
    1.58 with ids -1/-1, order 0 and execute on load 1."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        _row_widget(panel, "description_string_table_id").setValue(43998)
        _row_widget(panel, "short_description_string_table_id").setValue(0)
        _row_widget(panel, "description_order").setValue(100)
        _row_widget(panel, "execute_on_load").setChecked(False)

        assert window.trigger_edits is not None
        assert window.trigger_edits.dirty_indices() == [0]
        out = tmp_path / "edited.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)

        trigger = parse_triggers(load_map_and_units(out)).triggers[0]
        assert (
            trigger.description_stid,
            trigger.short_description_stid,
            trigger.description_order,
            trigger.execute_on_load,
        ) == (43998, 0, 100, 0)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_header_field_widgets_show_unset_order_and_tooltips_as_specified() -> None:
    """Description Order is a u32 with no sentinel: floor 0, no "(unset)"
    text. A string table id of -1 reads "(unset)". The order row's label
    carries its tooltip."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        order = _row_widget(panel, "description_order")
        assert (order.minimum(), order.specialValueText(), order.text()) == (0, "", "0")
        stid = _row_widget(panel, "description_string_table_id")
        assert (stid.minimum(), stid.text()) == (-1, "(unset)")
        label = panel.property_form.labelForField(order)
        assert label is not None
        assert "higher numbers are listed first" in label.toolTip()
        assert order.toolTip() == label.toolTip()
        assert window.trigger_edits is None, "building the form must record nothing"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
@pytest.mark.parametrize(
    ("name", "has_execute_on_load"),
    [("C2_ElCid_coop_1_v0_16.aoe2scenario", False), ("2_Joan_coop_2_v0_15.aoe2scenario", True)],
    ids=["1.41", "1.55"],
)
def test_execute_on_load_is_a_row_only_from_scenario_1_55(name: str, has_execute_on_load: bool) -> None:
    """GH #141: the library supports execute_on_load since 1.55, and the form
    follows the open file's version."""
    path = Path(__file__).resolve().parent.parent / "examples" / name
    if not path.exists():
        pytest.skip(f"{name} is not in examples/")
    window = _window()
    try:
        window.load_scenario(path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        # By index: a grouped file's first top-level row is a section header.
        panel.select_trigger(0)
        names = [spec.name for spec, *_ in panel._rows]
        assert "description_order" in names
        assert ("execute_on_load" in names) is has_execute_on_load
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_editing_a_condition_field_marks_its_own_trigger() -> None:
    """The minimal-diff guarantee reaching through a condition: the edit is
    nested two levels down but dirties exactly one trigger."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        target = _select_first_condition(panel)

        spin = None
        for spec, kind, _index, widget in panel._rows:
            if kind == "condition" and spec.kind == "int":
                spin = widget
                break
        assert spin is not None, "a condition must expose at least one int field"
        spin.setValue(spin.value() + 5)

        assert window.trigger_edits is not None
        assert window.trigger_edits.dirty_indices() == [target]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selection_survives_a_rebuild() -> None:
    """show_scenario() is what undo/redo rebuilds through, and a 590-trigger
    list that jumps back to the top on every undo is unusable."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        if panel.tree.topLevelItemCount() < 2:
            pytest.skip("fixture has too few triggers to test selection")
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))

        panel.show_scenario(window.scenario)
        assert panel.current_trigger_index() == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_inert_armour_attack_side_is_a_label_whichever_side_it_is() -> None:
    """Both fields locked by the live-value rule have to present alike.

    Found by screenshot, like the overlap defect: a disabled QComboBox shows
    its first choice, so an inert armour_attack_class read "WONDER (0)" as
    though that were its value, while the int locked by the same rule said
    "(unset)". The fixture carries one effect of each sourcing, which is what
    makes both directions testable here.
    """
    from PyQt5.QtWidgets import QLabel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        effects = panel.entry_tree.topLevelItem(2)
        assert effects is not None and effects.childCount() >= 2, (
            "fixture must carry both an armour-attack-sourced and a quantity-sourced effect"
        )

        seen = set()
        for child in range(2):
            panel.entry_tree.setCurrentItem(effects.child(child))
            locked = {s.name for s, *_ in panel._rows if s.read_only}
            for spec, _kind, _index, widget in panel._rows:
                if spec.read_only:
                    assert isinstance(widget, QLabel), f"{spec.name} must render as a label"
            seen.add(frozenset(locked & {"quantity", "armour_attack_quantity", "armour_attack_class"}))

        assert seen == {
            frozenset({"quantity"}),
            frozenset({"armour_attack_quantity", "armour_attack_class"}),
        }, f"both sourcings must appear, got {seen}"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- 4d: catalog and document-reference reference-field widgets -------------


def test_a_catalog_presentation_shows_a_catalog_line_edit_with_the_resolved_name() -> None:
    """object_list_unit_id (UnitInfo) on the fixture's second trigger is 4 --
    ARCHER, per slice 0's combined-catalog fix."""
    from descape.constant_picker import CatalogLineEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))  # "Fixture: armour split"
        effects = panel.entry_tree.topLevelItem(2)
        panel.entry_tree.setCurrentItem(effects.child(0))
        widget = _row_widget(panel, "object_list_unit_id")
        assert isinstance(widget, CatalogLineEdit)
        assert widget.line_edit.text() == "ARCHER"
        assert widget.value() == 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_trigger_id_reference_shows_the_referenced_trigger_by_name() -> None:
    from PyQt5.QtWidgets import QComboBox

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(2))  # "Fixture: references"
        effects = panel.entry_tree.topLevelItem(2)
        panel.entry_tree.setCurrentItem(effects.child(0))  # Activate Trigger, trigger_id=0
        widget = _row_widget(panel, "trigger_id")
        assert isinstance(widget, QComboBox)
        assert widget.currentData() == 0
        assert widget.currentText() == "Fixture: setup (0)"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_variable_reference_shows_the_referenced_variable_by_name() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(3))  # "Fixture: variable"
        effects = panel.entry_tree.topLevelItem(2)
        panel.entry_tree.setCurrentItem(effects.child(0))  # Change Variable, variable=0
        widget = _row_widget(panel, "variable")
        assert widget.currentData() == 0
        assert widget.currentText() == "fixture_var (0)"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_unset_catalog_reference_shows_the_unset_placeholder() -> None:
    """The fixture's first trigger has a display_instructions effect with
    object_list_unit_id left at -1 -- the one real unset REFERENCE value the
    shipped fixture carries."""
    from descape.constant_picker import CatalogLineEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))  # "Fixture: setup"
        effects = panel.entry_tree.topLevelItem(2)
        panel.entry_tree.setCurrentItem(effects.child(0))
        widget = _row_widget(panel, "object_list_unit_id")
        assert isinstance(widget, CatalogLineEdit)
        assert widget.value() is None
        assert widget.line_edit.text() == ""
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_unset_document_reference_defaults_to_the_unset_row() -> None:
    """No shipped fixture entry leaves a TriggerId/VariableId field unset
    (every real one, either presentation, is a set reference), so this
    exercises the widget builder directly rather than fishing for a row that
    does not exist -- panel._manager() still needs a loaded document behind
    it, which _triggers_window() supplies."""
    from descape.trigger_fields import UNSET, FieldSpec

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        spec = FieldSpec(name="trigger_id", kind="reference", presentation="TriggerId")
        widget = panel._build_document_reference_widget(spec, [("effect", 0)], UNSET, True)
        assert widget.currentData() == UNSET
        assert widget.currentText() == "(unset)"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_setting_a_catalog_reference_through_the_widget_round_trips(tmp_path: Path) -> None:
    """The plan's verification item 4, for the catalog widget: typing a name
    through CatalogLineEdit must reach the model with the right id and still
    be there -- correctly, not just present -- after a real save and reopen."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))  # "Fixture: setup"
        effects = panel.entry_tree.topLevelItem(2)
        panel.entry_tree.setCurrentItem(effects.child(0))  # display_instructions, unset
        widget = _row_widget(panel, "object_list_unit_id")
        widget.line_edit.setText("Town Center")
        widget.line_edit.editingFinished.emit()

        assert window.trigger_edits is not None
        assert window.trigger_edits.dirty_indices() == [0]

        out = tmp_path / "edited.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)

        reloaded = parse_triggers(load_map_and_units(out))
        assert reloaded.triggers[0].effects[0].object_list_unit_id == 109
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- large ENUM picker --------------------------------------------------------
#
# "Fixture: armour split" (list index 1), first effect: object_attributes =
# 8 (ObjectAttribute, 147 members) and armour_attack_class = 3 (DamageClass,
# 50), both editable; its second effect locks armour_attack_class read-only.


def test_a_large_enum_gets_the_picker_and_a_small_one_keeps_its_combo() -> None:
    from PyQt5.QtWidgets import QComboBox

    from descape.value_picker import ValueLineEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_armour_split_effect(panel, 0)
        attributes = _row_widget(panel, "object_attributes")
        assert isinstance(attributes, ValueLineEdit)
        assert attributes.value() == 8
        assert attributes.line_edit.text() == "ARMOR (8)"
        assert isinstance(_row_widget(panel, "armour_attack_class"), ValueLineEdit)
        assert isinstance(_row_widget(panel, "operation"), QComboBox), "Operation has 5 members"
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
    finally:
        window.edit_history.mark_saved()
        window.close()


def _switch_attribute(panel, value: int) -> None:
    """Commit an object_attributes edit through its picker, then let the
    deferred form rebuild run."""
    from PyQt5.QtWidgets import QApplication

    widget = _row_widget(panel, "object_attributes")
    widget.line_edit.setText(str(value))
    widget.line_edit.editingFinished.emit()
    QApplication.processEvents()


def _editable_cluster(panel) -> set[str]:
    cluster = {"quantity", "quantity_float", "armour_attack_quantity", "armour_attack_class"}
    return {spec.name for spec, *_ in panel._rows if spec.name in cluster and not spec.read_only}


def test_an_attribute_switch_relocks_the_form_without_reselecting() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_armour_split_effect(panel, 0)
        assert _editable_cluster(panel) == {"armour_attack_quantity", "armour_attack_class"}

        _switch_attribute(panel, 0)  # HIT_POINTS
        assert _editable_cluster(panel) == {"quantity"}
        assert _row_widget(panel, "quantity").value() == 2, "the amount carried, not the packed 196610"
        assert panel._current_entry()[:2] == ("effect", 0), "the same entry stays selected"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_float_attribute_gets_an_editable_float_row_with_the_real_value() -> None:
    from PyQt5.QtWidgets import QDoubleSpinBox, QLabel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_armour_split_effect(panel, 1)  # HIT_POINTS, quantity 45
        assert isinstance(_row_widget(panel, "quantity_float"), QLabel)

        _switch_attribute(panel, 13)  # WORK_RATE
        assert _editable_cluster(panel) == {"quantity_float"}
        row = _row_widget(panel, "quantity_float")
        assert isinstance(row, QDoubleSpinBox) and row.value() == 45.0

        row.setValue(2.5)
        effect = window.trigger_edits.manager().triggers[1].effects[1]
        assert effect.quantity == 2.5
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_stored_zero_shows_in_the_quantity_float_row(tmp_path: Path) -> None:
    """The library's reader swaps a stored 0.0 for the int half, here -1."""
    from PyQt5.QtWidgets import QDoubleSpinBox
    from test_effect_quantity import SHAPE_TRIGGER, shape_index, write_quantity_shapes

    window = _window()
    try:
        window.load_scenario(write_quantity_shapes(tmp_path / "shapes.aoe2scenario"))
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        panel.select_trigger(SHAPE_TRIGGER)
        effects = panel.entry_tree.topLevelItem(2)
        panel.entry_tree.setCurrentItem(effects.child(shape_index("sentinel int, stored 0.0")))
        row = _row_widget(panel, "quantity_float")
        assert isinstance(row, QDoubleSpinBox) and row.value() == 0.0
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_attribute_family_shares_one_field_order_on_the_form_and_the_row() -> None:
    """GH #278, GH #275: For Class lays out as Modify Attribute does, its row
    text opens with the same field its form does, and the live cluster slot
    never moves a row."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        for_class = _appended_effects(
            window, _effect_type_id(panel, "modify_attribute_for_class"), [{"quantity": 5, "object_group2": 3}]
        )[0]
        modify = _appended_effects(window, _effect_type_id(panel, "modify_attribute"), [{}])[0]
        cluster = ["quantity", "quantity_float", "armour_attack_quantity", "armour_attack_class"]

        _select_entries(panel, ("effect", for_class))
        assert [spec.name for spec, *_ in panel._rows] == [
            *cluster, "object_group2", "object_type2", "source_player", "operation", "object_attributes", "message",
        ]
        entry = panel._manager().triggers[0].effects[for_class]
        assert panel._describe("effect", entry)[1].startswith("quantity=5")

        _select_entries(panel, ("effect", for_class), ("effect", modify))
        assert [spec.name for spec, *_ in panel._rows] == [
            *cluster, "source_player", "operation", "object_attributes", "message",
        ]

        _select_armour_split_effect(panel, 0)
        before = [spec.name for spec, *_ in panel._rows]
        assert before[:4] == cluster
        _switch_attribute(panel, 13)  # WORK_RATE
        assert _editable_cluster(panel) == {"quantity_float"}
        assert [spec.name for spec, *_ in panel._rows] == before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_deferred_rebuild_skips_a_since_changed_selection() -> None:
    """The user can click away before the deferred rebuild runs."""
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_armour_split_effect(panel, 0)
        widget = _row_widget(panel, "object_attributes")
        widget.line_edit.setText("0")
        widget.line_edit.editingFinished.emit()
        _select_armour_split_effect(panel, 1)
        rows_before = [w for *_, w in panel._rows]
        QApplication.processEvents()
        assert [w for *_, w in panel._rows] == rows_before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_read_only_large_enum_still_renders_as_a_disabled_label() -> None:
    from PyQt5.QtWidgets import QLabel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_armour_split_effect(panel, 1)
        widget = _row_widget(panel, "armour_attack_class")
        assert isinstance(widget, QLabel) and not widget.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_enum_picker_shows_unset_as_unknown_and_none_as_the_placeholder() -> None:
    """Trap 5: -1 is not special-cased for enums (the combo showed
    "unknown (-1)"); only None, an unreachable attribute, is empty."""
    from descape.trigger_fields import ENUM, UNSET, FieldSpec, enum_choices

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_armour_split_effect(panel, 0)
        spec = FieldSpec("object_attributes", ENUM, enum_choices("ObjectAttribute"), presentation="ObjectAttribute")
        unset = panel._build_enum_picker_widget(spec, [("effect", 0)], UNSET, True)
        assert unset.value() == UNSET
        assert unset.line_edit.text() == "unknown (-1)"
        missing = panel._build_enum_picker_widget(spec, [("effect", 0)], None, True)
        assert missing.value() is None
        assert missing.line_edit.text() == ""
        assert missing.line_edit.placeholderText() == "(unset)"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_unchanged_enum_picker_commit_records_nothing() -> None:
    """A focus-out on untouched text must not push a phantom undo step."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_armour_split_effect(panel, 0)
        widget = _row_widget(panel, "object_attributes")
        widget.line_edit.editingFinished.emit()
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    ("typed", "expected", "shown"),
    [
        ("base melee", 4, "BASE MELEE (4)"),
        ("WAR ELEPHANTS (5)", 5, "WAR ELEPHANTS (5)"),
        ("9999", 9999, "unknown (9999)"),
    ],
    ids=["bare label", "rendered label", "out of vocabulary"],
)
def test_setting_a_large_enum_through_the_picker_round_trips(
    tmp_path: Path, typed: str, expected: int, shown: str
) -> None:
    """armour_attack_class (DamageClass, 50), not object_attributes: moving
    an armour-split effect's object_attributes off ARMOR rewrites the quantity
    cluster, which the cluster tests cover. A value the enum does not cover
    must survive untouched, via the raw-integer escape hatch."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_armour_split_effect(panel, 0)
        widget = _row_widget(panel, "armour_attack_class")
        widget.line_edit.setText(typed)
        widget.line_edit.editingFinished.emit()

        assert window.trigger_edits is not None
        assert window.trigger_edits.dirty_indices() == [1]

        out = tmp_path / "edited.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded = parse_triggers(load_map_and_units(out))
        assert reloaded.triggers[1].effects[0].armour_attack_class == expected
    finally:
        window.edit_history.mark_saved()
        window.close()

    reopened = _window()
    try:
        reopened.load_scenario(out)
        reopened.mode_combo.setCurrentText("Triggers")
        _select_armour_split_effect(reopened.trigger_panel, 0)
        assert _row_widget(reopened.trigger_panel, "armour_attack_class").line_edit.text() == shown
    finally:
        reopened.edit_history.mark_saved()
        reopened.close()


def test_setting_a_document_reference_through_the_widget_round_trips(tmp_path: Path) -> None:
    """The plan's verification item 4, for the document combo: picking a
    different trigger through the TriggerId combo must reach the model and
    survive a real save and reopen."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(2))  # "Fixture: references"
        effects = panel.entry_tree.topLevelItem(2)
        panel.entry_tree.setCurrentItem(effects.child(0))  # activate_trigger, trigger_id=0
        widget = _row_widget(panel, "trigger_id")
        target_index = widget.findData(1)  # "Fixture: armour split"
        assert target_index >= 0
        widget.setCurrentIndex(target_index)

        assert window.trigger_edits is not None
        assert window.trigger_edits.dirty_indices() == [2]

        out = tmp_path / "edited.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)

        reloaded = parse_triggers(load_map_and_units(out))
        target = next(t for t in reloaded.triggers if t.name == "Fixture: references")
        assert target.effects[0].trigger_id == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_trigger_id_reference_never_shows_a_stale_id_after_a_renumbering_delete() -> None:
    """The plan's trap 1: deleting a trigger renumbers trigger_id across the
    whole list, so a TriggerId combo built before the delete could in
    principle keep showing an id that no longer means what it did.

    Not a live defect, confirmed empirically rather than assumed: every
    structural op that can renumber (new/copy/delete) takes the "panel"
    refresh tier (viewer_triggers.py's _after_trigger_edit()), a full
    show_scenario() rebuild that resets entry-level selection to the
    trigger root rather than preserving whichever condition/effect a
    TriggerId combo was showing. So nothing is ever left on screen holding
    a stale id -- either the combo is rebuilt fresh, or it isn't shown.
    Only move_up/move_down are lighter than a full rebuild, and they never
    renumber at all (test_no_reorder_or_move_triggers_call_is_introduced).
    """
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(2))  # "Fixture: references"
        effects = panel.entry_tree.topLevelItem(2)
        panel.entry_tree.setCurrentItem(effects.child(1))  # Deactivate Trigger, trigger_id=1
        before_widget = _row_widget(panel, "trigger_id")
        assert before_widget.currentData() == 1

        window.trigger_structural_edit("delete", [0])  # delete "Fixture: setup" (id 0)

        assert panel._selected_entry_ref() is None, "selection resets to the trigger root, not a stale entry"
        assert not any(spec.name == "trigger_id" for spec, *_ in panel._rows), (
            "no reference widget survives the rebuild to be stale"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_reselecting_leaves_no_stale_widgets_behind() -> None:
    """Found by screenshotting the panel, not by any assertion.

    deleteLater() defers destruction to the next event-loop pass, so the
    previous entry's rows kept their geometry and the new ones drew on top:
    every label overlapping every other label. The form's own count() looked
    perfectly correct throughout, which is why this checks the host's children
    rather than the layout.
    """
    from PyQt5.QtWidgets import QWidget

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_first_condition(panel)
        effects = panel.entry_tree.topLevelItem(2)
        assert effects is not None and effects.childCount(), "fixture must carry an effect"

        panel.entry_tree.setCurrentItem(effects.child(0))
        # Counted against the form rather than against _rows: addRow() parents a
        # label widget per row as well as the field widget, so the form's own
        # count is the honest total of what should be on screen.
        parented = [
            child
            for child in panel.property_host.findChildren(QWidget)
            if child.parent() is panel.property_host
        ]
        assert len(parented) == panel.property_form.count(), (
            f"{len(parented) - panel.property_form.count()} widgets from a previous entry "
            f"are still parented and will draw over the current rows"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_editing_is_gated_on_the_write_support_flag() -> None:
    """Gated at panel-populate time, never at file-open time: the flag is only
    meaningful once parse_triggers() has run (4a deviation 1)."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        _select_first_condition(panel)

        assert not panel._editable
        for spec, _kind, _index, widget in panel._rows:
            # A read_only spec is a QLabel whatever the file allows, so the
            # meaningful assertion for those is that they are not editors.
            assert not widget.isEnabled(), f"{spec.name} must be disabled on a read-only file"

        spec = next(s for s, *_ in panel._rows if s.kind == "int")
        panel._changed(spec, [("condition", 0)], 4242)
        assert window.trigger_edits is None, "a gated panel must not reach the model"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_both_panes_scroll_instead_of_eliding() -> None:
    """All three calls together, or the text is cut with a scrollbar present.
    This is the "s..."/"e..." defect 4a.3 hit, and it is invisible to any
    assertion that only reads item text."""
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        for tree in (window.trigger_panel.tree, window.trigger_panel.entry_tree):
            assert not tree.header().stretchLastSection()
            assert tree.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded
            assert tree.textElideMode() == Qt.ElideNone
        assert window.trigger_panel.property_area.widgetResizable()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selecting_a_long_row_does_not_yank_the_horizontal_scroll() -> None:
    """Regression for the "selecting a long row yanks horizontal scroll" TODO
    item. Clicking a long Detail cell sets that column current, and Qt's
    default scrollTo(index, EnsureVisible) then scrolls until that cell's
    far-right edge is visible -- yanking the horizontal scrollbar away from
    wherever the user had it. _HScrollStableTreeWidget restores the
    horizontal position scrollTo() moved, while leaving the normal vertical
    auto-scroll (needed for keyboard navigation) untouched.
    """
    from PyQt5.QtWidgets import QApplication, QTreeWidgetItem

    from descape.trigger_panel import TriggerPanel
    from descape.value_picker import _HScrollStableTreeWidget

    conftest.ensure_qapp()
    tree = _HScrollStableTreeWidget()
    tree.setHeaderLabels(["Item", "Detail"])
    tree.setColumnCount(2)
    TriggerPanel._configure_scrolling(tree)

    top = QTreeWidgetItem(["a", "short"])
    tree.addTopLevelItem(top)
    for i in range(1, 50):
        tree.addTopLevelItem(QTreeWidgetItem([str(i), "short"]))
    long_item = QTreeWidgetItem(["z", "x" * 400])
    tree.addTopLevelItem(long_item)

    tree.setColumnWidth(1, 3000)
    tree.resize(120, 100)
    tree.show()
    QApplication.processEvents()

    tree.setCurrentItem(top, 1)
    QApplication.processEvents()
    assert tree.horizontalScrollBar().value() == 0
    assert tree.verticalScrollBar().value() == 0

    tree.setCurrentItem(long_item, 1)
    QApplication.processEvents()
    assert tree.horizontalScrollBar().value() == 0, (
        "selecting the long row's Detail cell must not move the horizontal scrollbar"
    )
    assert tree.verticalScrollBar().value() > 0, (
        "vertical auto-scroll to the selected row must still work"
    )


def test_a_trigger_tech_field_hides_internal_techs_and_an_object_field_does_not() -> None:
    from descape import object_catalog, trigger_fields
    from descape.constant_picker import HIDDEN_LABEL, INTERNAL_TECHS_LABEL

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        current = next(e for e in object_catalog.techs() if e.internal)
        tech = panel._build_catalog_widget(None, [], current.id, trigger_fields.CATALOG_PRESENTATIONS["TechInfo"], True)
        assert tech._hidden_label == INTERNAL_TECHS_LABEL
        assert {i.value for i in tech._items if i.hidden} == {e.id for e in object_catalog.techs() if e.internal}
        assert tech.line_edit.text() == current.name, "an internal tech already in a trigger still displays"
        unit = panel._build_catalog_widget(None, [], None, trigger_fields.CATALOG_PRESENTATIONS["UnitInfo"], True)
        assert unit._hidden_label == HIDDEN_LABEL
    finally:
        window.edit_history.mark_saved()
        window.close()
