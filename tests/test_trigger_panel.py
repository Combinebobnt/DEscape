"""Coverage for Triggers mode, the trigger browser (phase 4a.3), and its
property editor (phase 4b.6).

Same offscreen-ViewerWindow technique as tests/test_keybinds.py
(QT_QPA_PLATFORM=offscreen, one shared QApplication via conftest.ensure_qapp()).

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

import os
from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"
TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """No test here wants the real install, and AOE2DE_INSTALL_PATH outranks
    the config conftest hides, so a shell exporting it would leak one in."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _window():
    """A shown, fixed-size offscreen ViewerWindow.

    show() + processEvents() is load-bearing for anything reading the
    splitter: QSplitter.setSizes() is renormalized against the widget's real
    geometry, so on an unshown window every size assertion tests Qt's layout
    fallback rather than the code under test.

    Also releases any modifier an earlier test's QTest.keyClick left held:
    the multi-select tree reads the live modifiers on a plain setCurrentItem(),
    exactly as a click does, so a stale Ctrl turns it into a toggle.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    window = conftest.shown_window(1500, 900)
    for key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
        QTest.keyRelease(window, key)
    return window


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
        window.content_splitter.setSizes([200, 1000])
        window.mode_combo.setCurrentText("Triggers")
        assert window.content_splitter.sizes()[0] >= TriggerPanel.MIN_USEFUL_WIDTH
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_widening_never_shrinks_a_wider_user_chosen_pane() -> None:
    """The splitter position persists, so a width the user dragged to must
    survive a mode switch -- _widen_left_column only ever grows."""
    window = _window()
    try:
        window.content_splitter.setSizes([900, 600])
        window.mode_combo.setCurrentText("Triggers")
        assert window.content_splitter.sizes()[0] == 900
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


def _triggers_window():
    """A window in Triggers mode on the trigger-bearing fixture."""
    window = _window()
    window.load_scenario(TRIGGER_FIXTURE)
    window.mode_combo.setCurrentText("Triggers")
    return window


def _walk_every_row(panel) -> None:
    """Select every trigger, then every row of its detail tree. This is what a
    user does by clicking around, and it must record nothing.

    Walks both a top-level item and its children (4c: a grouped tree's
    sections nest real trigger rows under a header, or under the one
    synthetic "(before the first section)" row, which is skipped -- it isn't
    a trigger and Qt.ItemIsSelectable is off for it). On the flat, unshipped
    default fixture every top-level item has zero children, so this walks
    exactly as it did before 4c existed.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    def select(item) -> None:
        panel.tree.setCurrentItem(item)
        QApplication.processEvents()
        for t in range(panel.entry_tree.topLevelItemCount()):
            top = panel.entry_tree.topLevelItem(t)
            panel.entry_tree.setCurrentItem(top)
            QApplication.processEvents()
            for c in range(top.childCount()):
                panel.entry_tree.setCurrentItem(top.child(c))
                QApplication.processEvents()

    for i in range(panel.tree.topLevelItemCount()):
        top = panel.tree.topLevelItem(i)
        if top.data(0, Qt.UserRole) is not None:
            select(top)
        for c in range(top.childCount()):
            select(top.child(c))


def _trigger_row_count(tree) -> int:
    """Every real trigger row in `tree`, top-level or nested -- the count
    the status line reports, which diverges from topLevelItemCount() once
    4c's grouping is in play."""
    from PyQt5.QtCore import Qt

    total = 0
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        if top.data(0, Qt.UserRole) is not None:
            total += 1
        total += top.childCount()
    return total


def _select_first_condition(panel) -> int:
    """Select the first condition the fixture carries, and return its trigger
    index. Asserts rather than skips: a fixture with no conditions would make
    every entry-level test below vacuous while still reporting green."""
    for i in range(panel.tree.topLevelItemCount()):
        panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
        conditions = panel.entry_tree.topLevelItem(1)
        if conditions is not None and conditions.childCount():
            panel.entry_tree.setCurrentItem(conditions.child(0))
            return i
    raise AssertionError("the fixture carries no conditions, so entry tests cannot run")


def _row_widget(panel, field: str):
    """The live widget for one field of whatever the detail tree has selected."""
    for spec, _kind, _index, widget in panel._rows:
        if spec.name == field:
            return widget
    raise AssertionError(f"no {field} row in {[s.name for s, *_ in panel._rows]}")


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


def _select_armour_split_effect(panel, child: int) -> None:
    panel.select_trigger(1)
    panel.entry_tree.setCurrentItem(panel.entry_tree.topLevelItem(2).child(child))


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
    refresh tier (viewer.py's _after_trigger_edit()), a full
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

    from descape.trigger_panel import TriggerPanel, _HScrollStableTreeWidget

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


# -- 4b.6b: structural editing and the vocabulary picker ---------------------


def _group(panel, kind: str):
    """The Conditions or Effects heading row, found by its marker.

    By marker rather than by topLevelItem(1)/(2): those positions are right
    today and wrong the moment the detail tree gains a row.
    """
    from PyQt5.QtCore import Qt

    for i in range(panel.entry_tree.topLevelItemCount()):
        item = panel.entry_tree.topLevelItem(i)
        if item.data(0, Qt.UserRole) == ("group", kind):
            return item
    raise AssertionError(f"no {kind} group row in the detail tree")


def _pick(panel, kind: str, name: str) -> None:
    """Select one row of the open picker by its displayed name."""
    from PyQt5.QtCore import Qt

    for i in range(panel.picker_tree.topLevelItemCount()):
        group = panel.picker_tree.topLevelItem(i)
        for c in range(group.childCount()):
            child = group.child(c)
            if child.data(0, Qt.UserRole)[0] == kind and child.text(0) == name:
                panel.picker_tree.setCurrentItem(child)
                return
    raise AssertionError(f"no {kind} named {name!r} in the picker")


def test_opening_and_cancelling_the_picker_records_nothing(tmp_path: Path) -> None:
    """6b's own "browsing is not editing". commit_trigger_edit() pushes
    unconditionally, so the window must be told about an accepted pick and
    nothing else -- not about the button click that opened the picker.
    """
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))

        panel._request_entry_op("new")
        assert panel.detail_stack.currentIndex() == 1, "the picker page did not come up"
        # Browsing the picker is not editing either: select a row, filter, and
        # only then cancel.
        _pick(panel, "effect", "send chat")
        panel.picker_filter.setText("chat")
        panel._close_picker()

        assert panel.detail_stack.currentIndex() == 0
        assert window.trigger_edits is None, "opening the picker must not build an edit model"
        assert not window.edit_history.is_dirty
        assert not window.windowTitle().startswith("*")

        out = tmp_path / "cancelled.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_offers_both_kinds_and_filters_across_them() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        panel._request_entry_op("new")

        conditions = panel.picker_tree.topLevelItem(0)
        effects = panel.picker_tree.topLevelItem(1)
        assert "Conditions" in conditions.text(0) and "Effects" in effects.text(0)
        # A floor rather than an exact count, which a library bump would break
        # for no real reason: docs/INGAME_EDITOR_REFERENCE.md's traversal found
        # 40 conditions and 100 effects, and newer versions only add.
        assert conditions.childCount() >= 40
        assert effects.childCount() >= 100

        # But type 0, the library's "none" placeholder, is not offered: the
        # in-game editor lists neither, and alphabetically it would sit at the
        # top of both groups.
        names = [
            group.child(c).text(0)
            for group in (conditions, effects)
            for c in range(group.childCount())
        ]
        assert "none" not in names
        assert len(names) == conditions.childCount() + effects.childCount()
        assert f"({conditions.childCount()})" in conditions.text(0), (
            "the heading count must match what the group actually offers"
        )

        panel.picker_filter.setText("timer")
        visible = [
            group.child(c).text(0)
            for group in (conditions, effects)
            for c in range(group.childCount())
            if not group.child(c).isHidden()
        ]
        assert visible, "a filter matching real types hid everything"
        assert all("timer" in name for name in visible)

        # A group left with nothing under it hides too, rather than sitting
        # there as an empty heading.
        panel.picker_filter.setText("zzz-no-such-type-zzz")
        assert conditions.isHidden() and effects.isHidden()
        assert panel.picker_add_button.isEnabled() is False
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_collapses_the_other_kind_when_one_group_is_highlighted() -> None:
    """GH #136: New with the Effects (or Conditions) heading, or one of its
    rows, highlighted opens that kind's group and collapses the other."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        effects = _group(panel, "effect")
        assert effects.childCount() > 0, "fixture assumption: trigger 0 has an effect"

        def picker_expanded() -> list[bool]:
            groups = [panel.picker_tree.topLevelItem(i) for i in range(panel.picker_tree.topLevelItemCount())]
            assert [g.text(0).split(" ")[0] for g in groups] == ["Conditions", "Effects"]
            return [g.isExpanded() for g in groups]

        def open_from(item) -> list[bool]:
            panel._set_current_entry_item(item)
            panel._request_entry_op("new")
            expanded = picker_expanded()
            panel._close_picker()
            return expanded

        assert open_from(_group(panel, "effect")) == [False, True]
        assert open_from(_group(panel, "condition")) == [True, False]
        assert open_from(_group(panel, "effect").child(0)) == [False, True]
        assert open_from(panel.entry_tree.topLevelItem(0)) == [True, True], "the Trigger row opens both"

        # A filter opens both so a collapsed group cannot hide a match; clearing restores.
        panel._set_current_entry_item(_group(panel, "effect"))
        panel._request_entry_op("new")
        panel.picker_filter.setText("timer")
        assert picker_expanded() == [True, True]
        panel.picker_filter.clear()
        assert picker_expanded() == [False, True]
        panel._close_picker()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_adds_to_the_trigger_it_was_opened_against() -> None:
    """The target is latched at open. The trigger tree stays live while the
    picker shows, so reading the selection back at accept time would put the
    condition on whichever trigger the user had clicked in the meantime.
    """
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        before_0 = _group(panel, "condition").childCount()

        panel._request_entry_op("new")
        # The user clicks another trigger while the picker is up.
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        _pick(panel, "condition", "timer")
        panel._accept_pick()

        from descape import library_compat

        manager = window.trigger_edits.manager()
        assert len(manager.triggers[0].conditions) == before_0 + 1, "the pick missed its trigger"
        vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
        added = manager.triggers[0].conditions[-1]
        assert vocabulary.conditions[added.condition_type].name == "timer"
        # Only the latched trigger's blob is dirty; the one selected in the
        # meantime still splices its original bytes.
        assert window.trigger_edits.is_dirty(0)
        assert not window.trigger_edits.is_dirty(1)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_new_condition_gets_this_documents_own_vocabulary_defaults() -> None:
    """The library builds a new entry from its module-level default_attributes,
    which _initialise_version_dependencies rewrites per load. This pins that
    what comes out matches the vocabulary JSON for the document's own version,
    which is what the property form is about to render it with.
    """
    from descape import library_compat

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)

        window.entry_structural_edit("new", 0, "condition", -1, 3)
        window.entry_structural_edit("new", 0, "effect", -1, 55)
        manager = window.trigger_edits.manager()

        for kind, entries, created in (
            ("condition", vocabulary.conditions, manager.triggers[0].conditions[-1]),
            ("effect", vocabulary.effects, manager.triggers[0].effects[-1]),
        ):
            definition = entries[3 if kind == "condition" else 55]
            # Over `attributes`, the fields this type actually displays, not
            # over every field in default_attributes: the library normalises
            # the armour/attack list slots to None on a type that does not use
            # them, and those are never shown for such a type anyway.
            for attribute in definition.attributes:
                assert getattr(created, attribute) == definition.default_attributes[attribute], (
                    f"new {kind} {definition.name}: {attribute} is not the vocabulary default"
                )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_new_triggers_do_not_reuse_a_name_the_document_already_has() -> None:
    """Duplicate names are legal in-game, but 4c's folders will key on names --
    trigger_id does not survive a reorder -- so a new trigger gets a free one."""
    window = _triggers_window()
    try:
        for _ in range(3):
            window.trigger_structural_edit("new", ())
        names = [t.name for t in window.trigger_edits.manager().triggers]
        assert names[-3:] == ["New trigger", "New trigger 2", "New trigger 3"]
        assert len(set(names)) == len(names)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_sizes_its_column_to_its_longest_type_name() -> None:
    """Found by screenshotting the panel, not by an assertion: the picker's one
    column kept Qt's ~100 px default and clipped every row to "accumu",
    "ai signa", "bring o". ResizeToContents is safe on a single column where it
    is not on the two-column detail tree -- there is no second column for the
    longest row to push off the pane.
    """
    from PyQt5.QtWidgets import QHeaderView

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        panel._request_entry_op("new")

        assert panel.picker_tree.header().sectionResizeMode(0) == QHeaderView.ResizeToContents
        longest = max(
            (
                group.child(c).text(0)
                for i in range(panel.picker_tree.topLevelItemCount())
                for group in [panel.picker_tree.topLevelItem(i)]
                for c in range(group.childCount())
            ),
            key=len,
        )
        metrics = panel.picker_tree.fontMetrics()
        assert panel.picker_tree.columnWidth(0) >= metrics.horizontalAdvance(longest), (
            f"the picker column is narrower than {longest!r}, so it is clipping rows"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_picker_takes_the_whole_lower_pane_and_gives_it_back() -> None:
    """Also a screenshot finding: sharing the lower pane with the detail tree
    left 144 selectable types in five visible rows."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        assert panel.entry_tree.isVisible()

        panel._request_entry_op("new")
        assert not panel.entry_tree.isVisible()
        assert not panel.entry_new_button.isVisible()

        panel._close_picker()
        assert panel.entry_tree.isVisible()
        assert panel.entry_new_button.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_structural_buttons_track_the_selection_and_the_write_support_flag() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        assert panel.trigger_new_button.isEnabled()
        assert panel.trigger_copy_button.isEnabled()
        assert panel.entry_new_button.isEnabled()

        # The trigger row and the group headings are not entries, so there is
        # nothing for Copy/Delete to act on while one of them is selected.
        panel.entry_tree.setCurrentItem(panel.entry_tree.topLevelItem(0))
        assert not panel.entry_copy_button.isEnabled()
        assert not panel.entry_retype_button.isEnabled()
        # And the funnel refuses it too, not just the button: the trigger row
        # reaches _current_entry() as kind "trigger", where Copy and Delete go
        # through _selected_entry_ref(), which drops it.
        panel._request_entry_op("retype")
        assert panel.detail_stack.currentIndex() == 0, "the trigger row opened a retype picker"
        panel.entry_tree.setCurrentItem(_group(panel, "effect"))
        assert not panel.entry_copy_button.isEnabled()
        assert not panel.entry_delete_button.isEnabled()
        assert not panel.entry_retype_button.isEnabled()

        _select_first_condition(panel)
        assert panel.entry_copy_button.isEnabled()
        assert panel.entry_delete_button.isEnabled()
        assert panel.entry_retype_button.isEnabled()

        # And every one of them goes dead on a file that cannot be written.
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        _select_first_condition(panel)
        for button in (
            panel.trigger_new_button,
            panel.trigger_copy_button,
            panel.trigger_delete_button,
            panel.entry_new_button,
            panel.entry_copy_button,
            panel.entry_delete_button,
            panel.entry_retype_button,
        ):
            assert not button.isEnabled(), f"{button.text()} is live on a read-only file"

        # And the gate holds below the buttons too, not just on them.
        panel._request_trigger_op("new")
        panel._request_entry_op("new")
        panel._request_entry_op("retype")
        assert window.trigger_edits is None, "a gated panel must not reach the model"
        assert panel.detail_stack.currentIndex() == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- trigger reordering -------------------------------------------------------
#
# The shipped fixture holds an identity display order (4 triggers, 0..3), so
# every test above is blind to the bug this section exists to pin: the panel
# used to list triggers in raw list-index order (viewer.py's old
# `for index, trigger in enumerate(triggers)`), never consulting
# trigger_display_order at all. A non-identity order has to be forced in-test,
# the same technique tests/test_trigger_undo.py's
# test_undo_preserves_a_custom_display_order uses -- through the setter, as a
# *fresh* list (Trap 6: the library mutates a list handed to the setter in
# place, so reusing a variable this test still holds a reference to would
# corrupt it once the getter's list_changed side effect fires).


def _force_custom_display_order(window) -> list[int]:
    manager = window.trigger_panel._manager()
    custom = list(reversed(range(len(manager.triggers))))
    manager.trigger_display_order = list(custom)
    window.trigger_panel.show_scenario(window.scenario)
    return custom


def _tree_order(panel) -> list[int]:
    from PyQt5.QtCore import Qt

    return [panel.tree.topLevelItem(i).data(0, Qt.UserRole) for i in range(panel.tree.topLevelItemCount())]


def test_the_panel_sorts_by_display_order_by_default() -> None:
    """The regression 1a exists to prevent: without it, this shows [0,1,2,3]
    even though the file says otherwise."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        assert _tree_order(panel) == custom

        panel.sort_combo.setCurrentIndex(1)  # "File order (trigger ID)"
        assert _tree_order(panel) == list(range(len(custom)))

        panel.sort_combo.setCurrentIndex(0)  # back to "Display order"
        assert _tree_order(panel) == custom
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_current_trigger_index_returns_the_model_index_not_the_row() -> None:
    """The regression 1a exists to prevent, at the read side: under a
    non-identity display order the first *row* and the first *trigger* are
    different triggers."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        assert panel.current_trigger_index() == custom[0]
        assert custom[0] != 0, "the fixture must start at identity for this to bite"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selection_survives_a_sort_mode_change() -> None:
    """The same trigger stays selected across a sort-mode toggle, even though
    its row moves -- select_trigger() looks it up by index via
    _item_for_index, not by remembering a row number."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        target_index = custom[2]  # not at row 0 in either sort order
        panel.select_trigger(target_index)
        assert panel.current_trigger_index() == target_index

        panel.sort_combo.setCurrentIndex(1)
        assert panel.current_trigger_index() == target_index
        panel.sort_combo.setCurrentIndex(0)
        assert panel.current_trigger_index() == target_index
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selection_survives_a_panel_rebuild() -> None:
    """show_scenario()'s own same-document restore path, under a non-identity
    order: _selection_state()/_restore_selection() operate on rows and stay
    correct as long as the sort mode does not change mid-restore, which it
    does not here."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        target_index = custom[1]
        panel.select_trigger(target_index)

        panel.show_scenario(window.scenario)
        assert panel.current_trigger_index() == target_index
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_execution_order_readout_states() -> None:
    """The three states 1c defines, and specifically that showing them never
    builds a TriggerEditModel -- reading it through
    self._pending_option_values()'s lazily-constructed self.trigger_edits
    would violate the same contract
    test_browsing_every_row_saves_byte_identically pins."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        # The shipped fixture: check it actually carries a value one way,
        # then drive the other two states directly against exec_order_value's
        # contract rather than hunting for corpus files with every state.
        from descape.trigger_model import exec_order_value

        value = exec_order_value(window.scenario)
        assert value is not None, "fixture assumption: this file stores the flag"
        expected = (
            "Executes in trigger-ID order (legacy)." if value else "Executes in display order."
        )
        assert expected in panel.status.text()
        assert window.trigger_edits is None, "the readout must not build an edit model"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_pending_exec_order_flip_updates_the_triggers_readout() -> None:
    """The mirror image of what _pending_option_values() already solves for
    Map Options: flipping the flag there and switching to Triggers mode must
    show the *pending* mode, not the file's stored byte -- otherwise the
    status line asserts the old execution mode as fact."""
    window = _window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for("legacy_exec_order")
        before = window.map_options_panel.current_values()["legacy_exec_order"]
        widget.setCurrentIndex(widget.findData(1 - before))
        assert window.trigger_edits is not None, "fixture assumption: the flip built a model"

        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        expected = (
            "Executes in trigger-ID order (legacy) - unsaved change."
            if (1 - before)
            else "Executes in display order - unsaved change."
        )
        assert expected in panel.status.text(), panel.status.text()

        # GH #2: a second flip lands back on the stored byte, so nothing is unsaved.
        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for("legacy_exec_order")  # rebuilt on re-entry
        widget.setCurrentIndex(widget.findData(before))
        window.mode_combo.setCurrentText("Triggers")
        assert "unsaved change" not in panel.status.text(), panel.status.text()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_buttons_are_disabled_at_the_ends_of_display_order() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        count = panel.tree.topLevelItemCount()
        assert count >= 2, "fixture assumption: at least two triggers to move between"

        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        assert not panel.trigger_move_up_button.isEnabled()
        assert panel.trigger_move_down_button.isEnabled()

        panel.tree.setCurrentItem(panel.tree.topLevelItem(count - 1))
        assert panel.trigger_move_up_button.isEnabled()
        assert not panel.trigger_move_down_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_buttons_are_disabled_off_display_order_or_while_filtered() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        assert panel.trigger_move_up_button.isEnabled()

        panel.sort_combo.setCurrentIndex(1)  # File order
        assert not panel.trigger_move_up_button.isEnabled()
        assert not panel.trigger_move_down_button.isEnabled()
        assert "Display order" in panel.trigger_move_up_button.toolTip()
        panel.sort_combo.setCurrentIndex(0)

        panel.filter_edit.setText("Fixture")  # matches every shipped trigger
        assert not panel.trigger_move_up_button.isEnabled()
        assert "filter" in panel.trigger_move_up_button.toolTip()
        panel.filter_edit.clear()
        assert panel.trigger_move_up_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_raising_move_shows_the_rolled_back_order_not_a_phantom_success(monkeypatch) -> None:
    """The invariant this project holds for exactly this shape of failure: "a UI
    that catches such an error should say so rather than presenting the edit
    as a clean no-op" -- and presenting it as a clean *success* is worse.
    move_row() patches two rows blindly, without reading the model, purely
    from tree-row bounds; if structural_edit() raised for a *model*-level
    reason unrelated to those row bounds, move_row()'s own bounds check would
    not catch it, and patching the tree would show a swap that never actually
    happened in the model the next save reads from.

    A plain out-of-range move (moving trigger 0 up) does not reach this: both
    moved_display_order()'s bounds check and move_row()'s independently agree
    it is invalid, so move_row() itself already no-ops in that case -- this
    corrupts trigger_display_order directly (dropping index 1 from it, which
    the panel's tree has no way to know about) so structural_edit() raises
    (list.index() -> ValueError) for a reason move_row()'s own row-bounds
    check cannot see, and confirms *that* divergence is the one this guards.

    QMessageBox.warning() is monkeypatched, same as
    test_trigger_edit_contract.py's test_a_failed_edit_is_reported_and_still_recorded
    -- unpatched it raises a real modal that blocks forever offscreen.
    """
    from descape import viewer as viewer_module

    monkeypatch.setattr(viewer_module.QMessageBox, "warning", lambda *a, **k: None)
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        before = _tree_order(panel)
        assert before == [0, 1, 2, 3], "fixture assumption: identity order, 4 triggers"

        manager = window.trigger_panel._manager()
        corrupted = [0, 0, 2, 3]  # index 1 is gone; both row and target (1, 2) stay in-bounds
        manager.trigger_display_order = list(corrupted)

        window.trigger_structural_edit("move_down", [1])  # row 1 -> target 2, both valid rows

        model = window.trigger_edits
        assert list(model.manager().trigger_display_order) == corrupted, (
            "a raising move leaves the model's order exactly as it was -- "
            "structural_edit()'s except branch rolls back to the value it "
            "captured at entry, which is the already-corrupted array here"
        )
        # The buggy shape this test exists to catch would be [0, 2, 0, 3] or
        # similar: move_row(1, +1) blindly swapping rows 1 and 2 regardless of
        # what the model actually holds. The fix falls back to a full
        # repopulate on error, which re-reads the model and reports its real
        # (if corrupted) state honestly instead.
        assert _tree_order(panel) == corrupted, (
            "the tree must reflect the model's real order after a raising "
            "move, not a swap move_row() performed blindly"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_up_patches_one_row_without_a_full_repopulate() -> None:
    """The point of the "order" refresh tier: show_scenario() is not called,
    so the tree's own QTreeWidgetItem objects survive the move, not just
    their labels."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        before_items = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        moved_index = panel.current_trigger_index()

        window.trigger_structural_edit("move_up", [moved_index])

        assert _tree_order(panel)[0] == moved_index
        after_items = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        assert {id(i) for i in before_items} == {id(i) for i in after_items}, (
            "a full repopulate would have replaced every QTreeWidgetItem"
        )
        assert panel.current_trigger_index() == moved_index, "the moved trigger stays selected"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_up_then_move_down_returns_to_the_original_order() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        original = _tree_order(panel)
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        moved_index = panel.current_trigger_index()

        window.trigger_structural_edit("move_up", [moved_index])
        assert _tree_order(panel) != original
        window.trigger_structural_edit("move_down", [moved_index])
        assert _tree_order(panel) == original
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_moving_a_trigger_undoes_independently_of_an_exec_order_edit() -> None:
    """tests/test_options_undo.py's
    test_an_exec_order_edit_and_a_trigger_edit_undo_independently establishes
    this at the model layer for the exec-order axis; this is the same claim
    for a display-order move, exercised through the window."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        original = _tree_order(panel)
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        moved_index = panel.current_trigger_index()
        window.trigger_structural_edit("move_up", [moved_index])
        moved = _tree_order(panel)
        assert moved != original

        model = window._ensure_trigger_edits()
        assert model.exec_order_supported, "fixture assumption"
        with window._trigger_edit(model, "Set exec order") as m:
            m.set_exec_order(1 - m.exec_order)

        window.undo()  # undoes the exec-order flip only
        assert _tree_order(panel) == moved, "the move must survive undoing the later, unrelated edit"

        window.undo()  # undoes the move
        assert _tree_order(panel) == original
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- execution-position column ------------------------------------------------
#
# Column _COL_POS is always the trigger's 0-based position in
# trigger_display_order; only the header says whether that is also the
# execution position. See trigger_model.resolve_exec_mode().


def _positions(panel) -> list[str]:
    return [
        panel.tree.topLevelItem(i).text(panel._COL_POS) for i in range(panel.tree.topLevelItemCount())
    ]


def _header(panel) -> str:
    return panel.tree.headerItem().text(panel._COL_POS)


def test_the_position_column_reads_monotonically_under_display_sort() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        assert custom != sorted(custom), "the forced order must not be identity"
        assert _positions(panel) == [str(p) for p in range(len(custom))]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_position_column_is_display_position_under_file_order_sort() -> None:
    """Non-monotonic by design under File order: it is still each trigger's
    display position, not its row. Not a bug to "fix"."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        custom = _force_custom_display_order(window)
        panel.sort_combo.setCurrentIndex(1)  # "File order (trigger ID)"
        assert _tree_order(panel) == list(range(len(custom)))
        assert _positions(panel) == [str(custom.index(i)) for i in range(len(custom))]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    ("pending", "expected"), [(0, "Exec #"), (1, "Display #")], ids=["display-order file", "legacy file"]
)
def test_the_position_header_follows_the_execution_mode(pending: int, expected: str) -> None:
    """Driven through pending_exec_order, the same resolution a real stored
    byte goes through; the corpus test below covers stored values."""
    from descape.trigger_model import exec_order_value

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert exec_order_value(window.scenario) is not None, "fixture assumption: the flag is stored"
        panel.show_scenario(window.scenario, pending_exec_order=pending)
        assert _header(panel) == expected
        assert panel.tree.headerItem().toolTip(panel._COL_POS)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_pending_exec_order_flip_switches_the_position_header() -> None:
    """The column's counterpart to
    test_a_pending_exec_order_flip_updates_the_triggers_readout."""
    window = _window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        from descape.trigger_model import exec_order_value

        before = exec_order_value(window.scenario)
        assert _header(panel) == ("Display #" if before else "Exec #")

        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for("legacy_exec_order")
        widget.setCurrentIndex(widget.findData(1 - before))
        window.mode_combo.setCurrentText("Triggers")
        assert _header(panel) == ("Exec #" if before else "Display #")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_closing_the_document_resets_the_position_header() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.show_scenario(window.scenario, pending_exec_order=0)
        assert _header(panel) == "Exec #"
        panel.clear_document()
        assert _header(panel) == "Display #"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_up_relabels_both_swapped_rows() -> None:
    """The fast "order" tier never repopulates, so without move_row()'s own
    setText both rows keep their old numbers. Item identity is asserted too,
    so this cannot pass by regressing to a full repopulate."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        before_items = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        moved_index = panel.current_trigger_index()

        window.trigger_structural_edit("move_up", [moved_index])

        after_items = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        assert set(map(id, before_items)) == set(map(id, after_items))
        assert _positions(panel) == [str(p) for p in range(len(after_items))]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_grouped_move_keeps_a_pending_exec_order_flip() -> None:
    """move_row()'s grouped fallback repopulates, and must forward the pending
    flip: dropping it snaps header and readout back to the file's stored mode.
    The divider rename and the sort switch are the panel's other two repopulates."""
    from descape.trigger_model import exec_order_value

    window = _window()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        before = exec_order_value(window.scenario)
        window.mode_combo.setCurrentText("Map Options")
        widget = window.map_options_panel.widget_for("legacy_exec_order")
        widget.setCurrentIndex(widget.findData(1 - before))
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        assert panel.tree.topLevelItemCount() >= 3, "fixture assumption: room to group and still move"

        panel.select_trigger(panel._display_slots[0])
        name = _row_widget(panel, "name")
        name.setText("--- grouped move test ---")
        name.editingFinished.emit()
        assert panel._grouped, "fixture assumption: a divider-named trigger groups the tree"

        flipped = "Exec #" if before else "Display #"
        assert _header(panel) == flipped
        moved = panel._display_slots[2]
        panel.select_trigger(moved)
        window.trigger_structural_edit("move_up", [moved])

        assert panel._display_position[moved] == 1, "the move itself landed"
        assert _header(panel) == flipped
        assert "unsaved change" in panel.status.text(), panel.status.text()

        panel.sort_combo.setCurrentIndex(1)  # File order
        assert "unsaved change" in panel.status.text(), panel.status.text()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_the_position_column_matches_display_order_on_real_files(scenario_path) -> None:
    """Every row's position cell is its index in trigger_display_order, the
    header matches resolve_exec_mode(), and a grouped tree's synthetic header
    row has no position. Covers the not-stored case (F7_3_York) and the
    legacy + non-identity files by walking the whole corpus."""
    from PyQt5.QtCore import Qt

    from descape.trigger_model import exec_order_value, resolve_exec_mode

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        panel = window.trigger_panel
        display = list(panel._manager().trigger_display_order)
        mode = resolve_exec_mode(exec_order_value(window.scenario), None)[0]
        assert _header(panel) == panel._POSITION_HEADERS[mode][0]

        seen = 0
        for i in range(panel.tree.topLevelItemCount()):
            top = panel.tree.topLevelItem(i)
            for item in [top] + [top.child(c) for c in range(top.childCount())]:
                index = item.data(0, Qt.UserRole)
                if index is None:
                    assert item.text(panel._COL_POS) == ""
                    continue
                assert item.text(panel._COL_POS) == str(display.index(index))
                seen += 1
        assert seen == len(display)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_no_reorder_or_move_triggers_call_is_introduced() -> None:
    """Reordering here means permuting trigger_display_order only -- ids are
    never renumbered. A tests/test_private_api_guard.py-style source scan, so
    the ID-renumbering path (reorder_triggers()/move_triggers(), both
    display-order-resetting traps -- see trigger_model.py's structural_edit()
    docstring) cannot creep into trigger_structural_edit()'s move_up/
    move_down branch unnoticed."""
    import inspect

    from descape.viewer import ViewerWindow

    # The actual call shape (".reorder_triggers(" / ".move_triggers("), not a
    # bare word match -- this docstring itself names both methods by way of
    # explaining why they must not appear as calls.
    # GH #133's Section Up/Down permute display order in their own helper.
    source = inspect.getsource(ViewerWindow.trigger_structural_edit) + inspect.getsource(ViewerWindow._move_section)
    assert ".reorder_triggers(" not in source
    assert ".move_triggers(" not in source

    # GH #27 Trap 1: import_triggers() with any index but -1 routes through
    # move_triggers() -> reorder_triggers() inside the library, which a scan
    # for those two call shapes cannot see.
    from descape import trigger_clipboard

    clipboard_source = inspect.getsource(trigger_clipboard)
    for text in (source, clipboard_source):
        assert ".reorder_triggers(" not in text and ".move_triggers(" not in text
        for line in text.splitlines():
            if ".import_triggers(" in line:
                assert "index=-1)" in line, line
    assert ".import_triggers(" in clipboard_source


# -- column and form fitting -------------------------------------------------
#
# Every number here was measured on the shipped fixture at MIN_USEFUL_WIDTH
# before the fitting pass, and each one was a defect the panel shipped with.


def _select_an_effect(panel):
    """Select an effect with long field labels -- the worst case for the form."""
    from PyQt5.QtWidgets import QApplication

    for i in range(panel.tree.topLevelItemCount()):
        panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
        QApplication.processEvents()
        effects = _group(panel, "effect")
        if effects.childCount():
            panel.entry_tree.setCurrentItem(effects.child(0))
            QApplication.processEvents()
            return
    raise AssertionError("the fixture carries no effects")


def test_the_trigger_column_fits_its_names_instead_of_the_whole_pane() -> None:
    """It was pinned at MIN_USEFUL_WIDTH (340) while the longest name needed
    246, so a list of short names always carried a horizontal scrollbar with
    nothing to scroll to. Three columns now: position and ID are asserted
    with >=, not ==, since ResizeToContents sizes a section to the header's
    own hint too, and both headers are wider than a one-digit fixture value --
    sizeHintForColumn() looks at row content only. Trigger keeps strict
    equality, unchanged from before the numeric columns existed."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        tree = panel.tree
        numeric = (panel._COL_POS, panel._COL_ID)
        for column in numeric:
            assert tree.columnWidth(column) >= tree.sizeHintForColumn(column)
        assert tree.columnWidth(panel._COL_NAME) == tree.sizeHintForColumn(panel._COL_NAME)
        if sum(tree.columnWidth(c) for c in numeric) + tree.sizeHintForColumn(
            panel._COL_NAME
        ) <= tree.viewport().width():
            assert not tree.horizontalScrollBar().isVisible(), (
                "names that fit must not produce a scrollbar"
            )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_id_column_shows_each_row_s_stable_list_index() -> None:
    """Column 0 renders the same index the row's Qt.UserRole already carries
    -- no new plumbing, just a second visible copy of an existing value. Also
    pins the deliberate choice that the free-text filter stays name-only: a
    needle that is a real row's id but appears in no name must match nothing,
    not fall back to an id match."""
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        tree = window.trigger_panel.tree
        assert tree.topLevelItemCount() > 0, "the fixture changed"
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            index = item.data(0, Qt.UserRole)
            assert index is not None, "the shipped fixture is flat -- no synthetic header expected"
            assert int(item.text(window.trigger_panel._COL_ID)) == index

        panel = window.trigger_panel
        needle = tree.topLevelItem(0).text(panel._COL_ID)
        panel.filter_edit.setText(needle)
        assert all(tree.topLevelItem(i).isHidden() for i in range(tree.topLevelItemCount())), (
            "an id-shaped needle must not match by id -- the filter only reads the name column"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_detail_column_is_reachable_and_keeps_a_minimum_share() -> None:
    """Two separate defects in one tree.

    Item was fixed at 240 of a 340 px pane, leaving Detail 98 px. Worse, Detail
    itself was left at Qt's 100 px default with stretchLastSection off, so a
    1263 px detail string was cut at 100 px *at every scroll position* rather
    than merely off-screen.
    """
    from descape.trigger_panel import TriggerPanel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        tree = panel.entry_tree

        assert tree.columnWidth(1) == tree.sizeHintForColumn(1), (
            "the detail column must be content-sized, or scrolling cannot reach its text"
        )
        visible_detail = tree.viewport().width() - tree.columnWidth(0)
        assert visible_detail >= TriggerPanel._MIN_DETAIL_WIDTH, (
            f"only {visible_detail} px of detail is visible before scrolling"
        )
        assert tree.columnWidth(0) <= tree.sizeHintForColumn(0), (
            "the item column must never exceed what its content needs"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_columns_re_fit_when_the_pane_is_resized() -> None:
    """The cap is a function of the viewport, so computing it only at the width
    the panel happened to be built at makes it meaningless."""
    from PyQt5.QtWidgets import QApplication

    from descape.trigger_panel import TriggerPanel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        tree = panel.entry_tree

        window.content_splitter.setSizes([240, 1260])
        QApplication.processEvents()
        narrow = tree.viewport().width() - tree.columnWidth(0)
        assert narrow >= TriggerPanel._MIN_DETAIL_WIDTH, (
            f"narrowing the pane left only {narrow} px of detail visible"
        )

        window.content_splitter.setSizes([600, 900])
        QApplication.processEvents()
        assert tree.columnWidth(0) == tree.sizeHintForColumn(0), (
            "a pane with room to spare must give the item column its full content width"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def _assert_form_rows_do_not_overlap(form) -> tuple[int, int]:
    """Walk every row, label-and-field, wrapped and full-width (spanning)
    alike: no widget squeezed below its minimum, no row drawing over the one
    above, and a wrapped row's editor below its own label. Returns (rows
    walked, rows wrapped). Rows with no widget are skipped; a spanning widget
    answers FieldRole too."""
    from PyQt5.QtWidgets import QFormLayout, QLabel

    walked = wrapped = 0
    previous_bottom = None
    for row in range(form.rowCount()):
        widgets = {}
        for role in (QFormLayout.LabelRole, QFormLayout.FieldRole, QFormLayout.SpanningRole):
            item = form.itemAt(row, role)
            if item is not None and item.widget() is not None and item.widget() not in widgets.values():
                widgets[role] = item.widget()
        if not widgets:
            continue
        walked += 1
        name = next((w.text() for w in widgets.values() if isinstance(w, QLabel)), f"row {row}")
        for widget in widgets.values():
            assert widget.height() >= widget.minimumSizeHint().height(), f"{name!r} is squeezed below its own minimum height"
        geometries = [widget.geometry() for widget in widgets.values()]
        if previous_bottom is not None:
            assert min(g.top() for g in geometries) >= previous_bottom, f"{name!r} draws over the row above it"
        label, field = widgets.get(QFormLayout.LabelRole), widgets.get(QFormLayout.FieldRole)
        if label is not None and field is not None and field.geometry().left() < label.geometry().right():
            wrapped += 1
            assert field.geometry().top() >= label.geometry().bottom(), f"{name!r}'s wrapped editor overlaps its label"
        previous_bottom = max(g.bottom() for g in geometries)
    return walked, wrapped


@pytest.mark.font_sensitive
@pytest.mark.parametrize("what", ["effect", "trigger"])
def test_a_long_form_scrolls_instead_of_crushing_its_rows(what: str) -> None:
    """Found by screenshot, and invisible to every other measurement: host
    width, scrollbar state and field widths all read correct while each row was
    squeezed to 6 px and drew over the next one.

    The cause is that a wrapped row's height depends on its width, and a
    QScrollArea sizes its widget from sizeHint(), computed as if nothing
    wrapped -- so the form was handed the unwrapped height and overflowed
    inside it. Asking the layout for heightForWidth, *after* activating it, is
    what makes the vertical scrollbar appear instead.

    The trigger form adds full-width caption rows and GH #141's long-label
    rows, which WrapLongRows puts on two lines (GH #139).
    """
    from PyQt5.QtWidgets import QApplication, QFormLayout

    from testkit.qt_window import FONT_DPI_OVERRIDE_ENV

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        if what == "effect":
            _select_an_effect(panel)
        else:
            panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
            QApplication.processEvents()
        form = panel.property_form
        assert len(panel._rows) >= 8, "this fixture no longer exercises a long form"
        walked, wrapped = _assert_form_rows_do_not_overlap(form)
        assert walked >= len(panel._rows)
        if what == "trigger":
            spanning = [r for r in range(form.rowCount()) if form.itemAt(r, QFormLayout.SpanningRole)]
            assert len(spanning) >= 4, "the two prose captions and boxes are full-width rows"
            if not os.environ.get(FONT_DPI_OVERRIDE_ENV):
                # At the pinned baseline font both string table id rows wrap; a smaller DPI fits them beside.
                assert wrapped >= 2, "the wrapped-row check walked nothing"

        assert panel.property_host.height() > panel.property_area.viewport().height(), (
            "a form taller than its pane must make the host taller, not compress the rows"
        )
        assert panel.property_area.verticalScrollBar().isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.font_sensitive
def test_dragging_the_splitter_refits_the_form_height() -> None:
    """A splitter drag reaches the form only through TriggerPanel.resizeEvent.
    The form's height is a function of its width (wrapped rows, and the
    Display Instructions preview's aspect), so the host's must follow it."""
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_an_effect(panel)
        form = panel.property_form
        needed = []
        for sizes in ([600, 900], [330, 1170], [900, 600]):
            window.content_splitter.setSizes(sizes)
            QApplication.processEvents()
            width = panel.property_area.viewport().width()
            needed.append(max(form.minimumSize().height(), form.heightForWidth(width)))
            assert panel.property_host.minimumHeight() == needed[-1], (sizes, width)
            _assert_form_rows_do_not_overlap(form)
        assert len(set(needed)) == len(needed), f"two widths need the same height, so a step checks nothing: {needed}"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.font_sensitive
def test_no_field_demands_more_width_than_the_pane_can_give() -> None:
    """ObjectAttribute's longest entry made its combo 430 px wide against a
    340 px panel, so the form sat permanently below its own stated minimum.
    That is what made every attempt to narrow the label column tip the whole
    form into horizontal overflow instead.

    Walks every row of every entry rather than sampling one. The offending
    combo is on a single effect of a single trigger, so a test that picked "an
    effect" passed while the defect was still there -- which is exactly what
    the first version of this test did.
    """
    from PyQt5.QtWidgets import QApplication

    from descape.trigger_panel import TriggerPanel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        checked = 0
        for i in range(panel.tree.topLevelItemCount()):
            panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
            QApplication.processEvents()
            for t in range(panel.entry_tree.topLevelItemCount()):
                top = panel.entry_tree.topLevelItem(t)
                for c in range(top.childCount()):
                    panel.entry_tree.setCurrentItem(top.child(c))
                    QApplication.processEvents()
                    # GH #138's Set / Go to / Reset rows live outside _rows.
                    groups = [
                        (f"{group} group row", row.set_button.parentWidget()) for group, row in panel._group_rows.items()
                    ]
                    if panel._instruction_preview is not None:  # GH #140's preview row, also outside _rows
                        groups.append(("instruction preview", panel._instruction_preview))
                    for name, widget in [(spec.name, widget) for spec, *_rest, widget in panel._rows] + groups:
                        checked += 1
                        assert (
                            widget.minimumSizeHint().width() <= TriggerPanel.MIN_USEFUL_WIDTH
                        ), (
                            f"{name}'s editor demands "
                            f"{widget.minimumSizeHint().width()} px of a "
                            f"{TriggerPanel.MIN_USEFUL_WIDTH} px panel"
                        )
                    assert (
                        panel.property_form.minimumSize().width() <= panel.property_host.width()
                    ), "the form's minimum width exceeds the space it is given"
                    assert not panel.property_area.horizontalScrollBar().isVisible()
        # 43 on the shipped fixture. A floor, so a fixture that lost its
        # entries cannot make this pass vacuously.
        assert checked >= 40, f"only {checked} fields walked -- the fixture changed"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_panel_populates_and_filters_a_real_scenario(scenario_path) -> None:
    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        tree = panel.tree

        if not window.scenario.trigger_read_supported:
            # The 1.54/trigger-3.9 set: an explanatory message, no rows, and
            # the filter disabled rather than a silently empty list.
            assert tree.topLevelItemCount() == 0
            assert "can't be read" in panel.status.text()
            assert not panel.filter_edit.isEnabled()
            return

        # The status line counts triggers, not tree rows -- these diverge
        # once 4c's grouping turns some rows into section headers with
        # children, so this counts every real trigger row, nested or not.
        count = _trigger_row_count(tree)
        assert f"{count} trigger" in panel.status.text()
        if count == 0:
            return

        all_items = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        all_items += [c for top in all_items for c in (top.child(j) for j in range(top.childCount()))]

        # A filter matching nothing hides every row; clearing restores them.
        panel.filter_edit.setText("zzz-no-such-trigger-zzz")
        assert all(item.isHidden() for item in all_items)
        panel.filter_edit.setText("")
        assert not any(item.isHidden() for item in all_items)

        # Filtering on a real trigger's own name keeps at least that one.
        # tree.topLevelItem(0) may be 4c's synthetic header (no real name of
        # its own), so pick the first item that actually carries a trigger.
        from PyQt5.QtCore import Qt

        real = next(item for item in all_items if item.data(0, Qt.UserRole) is not None)
        panel.filter_edit.setText(real.text(panel._COL_NAME))
        assert not real.isHidden()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_execution_order_readout_covers_the_no_stored_byte_case(scenario_path) -> None:
    """The third readout state -- "not stored" -- needs a real file to reach:
    F7_3_York is scenario 1.55 but its trigger version is below 4.5, so the
    exec-order byte consumes zero bytes despite the version implying
    otherwise. Not hardcoded to that
    filename -- any corpus file with the same shape exercises this the same
    way, and the corpus tier's own summary already reports skip counts, so a
    corpus with no such file reads as 100% skipped rather than silently
    green.
    """
    from descape.trigger_model import exec_order_value

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if exec_order_value(window.scenario) is not None:
            pytest.skip(f"{scenario_path.name}: this file does store the exec-order byte")

        assert "Execution order is not stored in this file." in window.trigger_panel.status.text()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- 4c: trigger organization (grouped tree, tag facet) -----------------------
#
# The shipped default-tier fixture has 4 triggers and no dividers, so it
# cannot exercise grouping at all -- see trigger_organize.py's own module
# docstring. These run at corpus tier instead, against real files whose
# authors actually used the `--- Section ---`/`[tag]` conventions this
# feature is built on. Generic over scenario_path rather than hardcoded to
# one corpus filename, with an explicit skip when a given file has nothing to
# exercise -- the corpus tier's own summary already reports skip counts, so
# an empty corpus reads as 100% skipped rather than silently green.


def _panel_sections(panel) -> list:
    from PyQt5.QtCore import Qt

    from descape.trigger_organize import Section

    result = []
    for i in range(panel.tree.topLevelItemCount()):
        top = panel.tree.topLevelItem(i)
        header_index = top.data(0, Qt.UserRole)
        members = tuple(top.child(c).data(0, Qt.UserRole) for c in range(top.childCount()))
        result.append(Section("", header_index, members))
    return result


@pytest.mark.corpus
def test_grouped_tree_matches_trigger_organize_sections_on_a_real_scenario(scenario_path) -> None:
    """Cross-checks the panel's own grouped-tree construction against an
    independent call to trigger_organize.sections() on the same manager data
    -- the panel must not silently diverge from the heuristic it's built on."""
    from descape.trigger_organize import sections as compute_sections

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        manager = panel._manager()
        if manager is None or not manager.triggers:
            pytest.skip(f"{scenario_path.name}: no triggers")

        names = [t.name or "" for t in manager.triggers]
        order = list(manager.trigger_display_order)
        expected = compute_sections(names, order)
        is_grouped = not (len(expected) == 1 and expected[0].header_index is None)

        assert panel._grouped == is_grouped
        if is_grouped:
            # Compare header/member indices only: the tree's own header text
            # comes from _trigger_label() (adds "(disabled)", substitutes
            # "(unnamed)"), not the raw name compute_sections() used.
            got = [(s.header_index, s.member_indices) for s in _panel_sections(panel)]
            want = [(s.header_index, s.member_indices) for s in expected]
            assert got == want
        else:
            assert _trigger_row_count(panel.tree) == panel.tree.topLevelItemCount()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_file_order_is_always_flat_on_a_real_scenario(scenario_path) -> None:
    """4c's grouping exists only under Display order -- switching to File
    order must flatten the tree even on a file whose names are full of
    dividers."""
    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if not panel.tree.topLevelItemCount():
            pytest.skip(f"{scenario_path.name}: no triggers")

        panel.sort_combo.setCurrentIndex(1)  # File order
        assert not panel._grouped
        for i in range(panel.tree.topLevelItemCount()):
            assert panel.tree.topLevelItem(i).childCount() == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_tag_facet_hides_non_matching_rows_on_a_real_scenario(scenario_path) -> None:
    """The tag combo matches the stashed _TAG_ROLE, never the rendered label
    -- see that constant's docstring for why inheriting the label-matching
    false-positive class into a facet would be worse here than in the
    free-text filter."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if panel.tag_combo.count() <= 1:
            pytest.skip(f"{scenario_path.name}: no [tag] names in this file")

        panel.tag_combo.setCurrentIndex(1)  # the first real tag, alphabetical
        wanted = panel.tag_combo.currentData()
        assert wanted is not None

        for i in range(panel.tree.topLevelItemCount()):
            top = panel.tree.topLevelItem(i)
            if top.isHidden():
                continue
            visible_children = [
                top.child(c) for c in range(top.childCount()) if not top.child(c).isHidden()
            ]
            header_matches = top.data(0, Qt.UserRole) is not None and wanted in top.data(0, panel._TAG_ROLE)
            # A visible row must owe its visibility to a real match -- its own
            # tag chain (flat rows and real section headers), or a visible child's.
            assert header_matches or visible_children
            for child in visible_children:
                assert wanted in child.data(0, panel._TAG_ROLE)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_renaming_across_the_divider_predicate_regroups_a_real_scenario(scenario_path) -> None:
    """4c's silent-staleness case 1: a rename that flips is_divider() must
    trigger a full repopulate, promoting the renamed trigger to its own
    section header -- a label patch alone would leave the tree structurally
    stale."""
    from descape.trigger_organize import is_divider

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if not panel._editable:
            pytest.skip(f"{scenario_path.name}: triggers are read-only")
        manager = panel._manager()
        if manager is None or not manager.triggers:
            pytest.skip(f"{scenario_path.name}: no triggers")

        ordinary = next((i for i, t in enumerate(manager.triggers) if not is_divider(t.name or "")), None)
        if ordinary is None:
            pytest.skip(f"{scenario_path.name}: every trigger name is already divider-shaped")

        panel.select_trigger(ordinary)
        name = _row_widget(panel, "name")
        name.setText("--- 4c test section ---")
        name.editingFinished.emit()

        assert panel._grouped
        renamed_item = panel._item_for_index[ordinary]
        assert renamed_item.parent() is None, "a divider-named trigger becomes its own section header"
        assert panel.current_trigger_index() == ordinary, "the renamed trigger stays selected"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_opening_a_grouped_real_scenario_selects_a_trigger_not_the_synthetic_header(scenario_path) -> None:
    """4c's silent-staleness case 2: opening a file with a before-first run of
    triggers must land on the first real trigger, not the synthetic
    "(before the first section)" header -- selecting the header would open
    the entry pane empty."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if not panel._grouped or not panel.tree.topLevelItemCount():
            pytest.skip(f"{scenario_path.name}: not grouped")
        first = panel.tree.topLevelItem(0)
        if first.data(0, Qt.UserRole) is not None:
            pytest.skip(f"{scenario_path.name}: no leading run before the first divider")

        assert first.text(panel._COL_ID) == "", "the synthetic header is not a trigger and has no id to show"
        assert first.text(panel._COL_POS) == "", "nor a display position"
        assert panel.current_trigger_index() is not None
        assert panel.tree.currentItem() is first.child(0)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_browsing_a_grouped_real_scenario_saves_byte_identically(scenario_path, tmp_path) -> None:
    """Extends test_browsing_every_row_saves_byte_identically's guarantee to
    grouped browsing: expanding/collapsing every section and switching the
    tag facet must never build a TriggerEditModel, same as plain browsing."""
    from descape.scenario_write import write_scenario

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if not panel._grouped:
            pytest.skip(f"{scenario_path.name}: no dividers, nothing to group")

        from PyQt5.QtWidgets import QApplication

        for i in range(panel.tree.topLevelItemCount()):
            top = panel.tree.topLevelItem(i)
            top.setExpanded(False)
            QApplication.processEvents()
            top.setExpanded(True)
            QApplication.processEvents()

        # The Sections menu: one jump per heuristic section, and collapse,
        # expand and every jump are browsing too.
        assert len(panel._section_actions) == len(panel._sections) == panel.tree.topLevelItemCount()
        panel.collapse_all_action.trigger()
        for action in panel._section_actions:
            action.trigger()
            QApplication.processEvents()
        panel.expand_all_action.trigger()

        for i in range(panel.tag_combo.count()):
            panel.tag_combo.setCurrentIndex(i)
            QApplication.processEvents()
        panel.tag_combo.setCurrentIndex(0)

        _walk_every_row(panel)

        assert window.trigger_edits is None, "browsing must not build an edit model"
        assert not window.edit_history.is_dirty, "browsing must record no undo step"

        out = tmp_path / scenario_path.name
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == scenario_path.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- XS script fields (P5-a) -------------------------------------------------


def _script_call_id(window, kind: str) -> int:
    from descape import library_compat

    vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
    entries = vocabulary.conditions if kind == "condition" else vocabulary.effects
    return next(e.id for e in entries.values() if e.name == "script_call")


def _plant_script_call(window, kind: str, stored: str) -> str:
    """Add a script_call to trigger 0 of the parsed manager, pre-model, holding
    `stored`, and select it. Returns the XS field's name. Not routed through
    the model on purpose: the assertion is that browsing it builds none."""
    panel = window.trigger_panel
    trigger = panel._manager().triggers[0]
    field = "xs_function" if kind == "condition" else "message"
    if kind == "condition":
        trigger._add_condition(_script_call_id(window, kind))
        entries = trigger.conditions
    else:
        trigger._add_effect(_script_call_id(window, kind))
        entries = trigger.effects
    setattr(entries[-1], field, stored)
    panel.select_trigger(0)
    panel.refresh_entries(select=(kind, len(entries) - 1))
    return field


@pytest.mark.parametrize("kind", ["condition", "effect"])
def test_a_script_call_field_is_a_multi_line_editor_with_its_lines(kind: str) -> None:
    from descape.text_edits import XsTextEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        field = _plant_script_call(window, kind, "void f()\r{\r  int a = 0;\r}")
        widget = _row_widget(panel, field)
        assert isinstance(widget, XsTextEdit)
        assert widget.toPlainText() == "void f()\n{\n  int a = 0;\n}"
        assert widget.document().blockCount() == 4
        assert window.trigger_edits is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("stored", ["a\rb", "a\r\nb", "a\nb"], ids=["CR", "CRLF", "LF"])
def test_an_untouched_xs_focus_out_records_nothing_whatever_the_separator(stored: str) -> None:
    """CRLF and LF are the cases _changed()'s equality check alone would miss:
    they display as "a\\nb" and translate back to "a\\rb", which differs from
    what is stored. Only the latch keeps a bare focus-out a no-op."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        field = _plant_script_call(window, "effect", stored)
        _row_widget(panel, field).editingFinished.emit()
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_typing_into_an_xs_field_records_nothing_until_focus_out() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        field = _plant_script_call(window, "effect", "a\rb")
        widget = _row_widget(panel, field)
        widget.appendPlainText("c")
        widget.appendPlainText("d")
        assert window.trigger_edits is None, "an edit must wait for editingFinished"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_editing_an_xs_field_writes_cr_separated_bytes(tmp_path: Path) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        window.entry_structural_edit("new", 0, "effect", -1, _script_call_id(window, "effect"))
        effect_index = len(window.trigger_edits.manager().triggers[0].effects) - 1
        panel.select_trigger(0)
        panel.refresh_entries(select=("effect", effect_index))
        steps = window.edit_history.cursor

        widget = _row_widget(panel, "message")
        widget.setPlainText("void f()\n{\n  // a comment\n  int a = 0;\n}")
        widget.editingFinished.emit()
        assert window.edit_history.cursor == steps + 1
        widget.editingFinished.emit()
        assert window.edit_history.cursor == steps + 1, "a second focus-out is not a second edit"

        out = tmp_path / "xs.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded = parse_triggers(load_map_and_units(out))
        assert reloaded.triggers[0].effects[effect_index].message == (
            "void f()\r{\r  // a comment\r  int a = 0;\r}"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_locked_xs_field_renders_its_lines() -> None:
    from PyQt5.QtWidgets import QLabel

    from descape.trigger_fields import STR, XS, FieldSpec

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_script_call(window, "effect", "a\rb")
        spec = FieldSpec("message", STR, sentinel=None, read_only=True, multiline=XS)
        label = panel._build_widget(spec, [("effect", 0)], panel._read(panel._current_entry()[2], spec.attribute))
        assert isinstance(label, QLabel)
        assert label.text() == "a\nb"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- prose fields ------------------------------------------------------------

# One long line with no newline in it: the dominant real shape of
# display_instructions.message, and what the XS widget renders as a single
# line behind a horizontal scrollbar.
_LONG_LINE = (
    "Defend the town centre until the timer runs out, then escort the "
    "relic cart to the monastery on the far side of the river before "
    "Attila's cavalry reaches the ford."
)


def _effect_id(window, name: str) -> int:
    from descape import library_compat

    vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
    return next(e.id for e in vocabulary.effects.values() if e.name == name)


def _plant_prose_effect(window, stored: str, name: str = "display_instructions") -> int:
    """Add a prose-message effect to trigger 0 of the parsed manager,
    pre-model, holding `stored`, and select it. Same deliberate no-model route
    as _plant_script_call()."""
    panel = window.trigger_panel
    trigger = panel._manager().triggers[0]
    trigger._add_effect(_effect_id(window, name))
    trigger.effects[-1].message = stored
    panel.select_trigger(0)
    panel.refresh_entries(select=("effect", len(trigger.effects) - 1))
    return len(trigger.effects) - 1


def _plant_trigger_description(window, stored: str) -> None:
    """The trigger's own description, which flows through the same
    _build_widget/_changed path as an effect field."""
    panel = window.trigger_panel
    panel._manager().triggers[0].description = stored
    panel.select_trigger(0)
    panel.refresh_entries()


def _plant_prose(window, where: str, stored: str):
    if where == "effect message":
        return _plant_prose_effect(window, stored)
    _plant_trigger_description(window, stored)
    return None


_PROSE_WHERE = ["effect message", "trigger description"]
_PROSE_FIELD = {"effect message": "message", "trigger description": "description"}


@pytest.mark.parametrize("where", _PROSE_WHERE)
@pytest.mark.font_sensitive
def test_a_prose_field_is_a_wrapping_multi_line_editor(where: str) -> None:
    """The assertion the XS tests deliberately do not make: a value with no
    newline at all still occupies several visual lines, with no horizontal
    scrollbar. GH #38 is exactly that case."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from descape.text_edits import ProseTextEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose(window, where, _LONG_LINE)
        widget = _row_widget(panel, _PROSE_FIELD[where])
        assert isinstance(widget, ProseTextEdit)
        assert widget.toPlainText() == _LONG_LINE
        QApplication.processEvents()
        assert widget.document().blockCount() == 1, "one paragraph, wrapped -- not split"
        assert widget.document().firstBlock().layout().lineCount() > 1, "did not wrap"
        # The policy, not isVisible(): every widget of a never-shown window
        # answers isVisible() False, so that assertion would hold for the
        # NoWrap widget too and prove nothing.
        assert widget.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert window.trigger_edits is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("where", _PROSE_WHERE)
def test_a_prose_field_shows_its_stored_lines(where: str) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose(window, where, "first line\r\nsecond line\r\nthird line")
        widget = _row_widget(panel, _PROSE_FIELD[where])
        assert widget.toPlainText() == "first line\nsecond line\nthird line"
        assert widget.document().blockCount() == 3
        assert widget.newline_token == "\r\n"
    finally:
        window.edit_history.mark_saved()
        window.close()


def _caption_geometry(panel, text: str):
    from PyQt5.QtWidgets import QLabel

    labels = [w for w in panel.property_host.findChildren(QLabel) if w.text() == text and w.isVisible()]
    assert len(labels) == 1, [w.text() for w in panel.property_host.findChildren(QLabel)]
    return labels[0].geometry()


@pytest.mark.parametrize(
    "where,field",
    [
        ("trigger", "description"),
        ("trigger", "short_description"),
        ("effect message", "message"),
        ("script call", "message"),
    ],
)
def test_a_multi_line_box_sits_full_width_under_its_caption(where: str, field: str) -> None:
    """GH #38: like Messages mode, the caption gets its own row and the box
    spans the form below it. One-line fields keep the side caption."""
    from PyQt5.QtWidgets import QApplication

    from descape.text_edits import _MultiLineEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        if where == "trigger":
            _plant_trigger_description(window, "some text")
        elif where == "effect message":
            _plant_prose_effect(window, "some text")
        else:
            _plant_script_call(window, "effect", "void f() {}")
        QApplication.processEvents()
        widget = _row_widget(panel, field)
        assert isinstance(widget, _MultiLineEdit)
        spec = next(s for s, *_ in panel._rows if s.name == field)
        form = panel.property_form.contentsRect()
        box = widget.geometry()
        assert (box.left(), box.right()) == (form.left(), form.right()), (box, form)
        caption = _caption_geometry(panel, spec.label)
        assert caption.left() == form.left()
        assert caption.bottom() < box.top(), "the caption sits above its box"

        if where == "trigger":
            name = _row_widget(panel, "name")
            name_caption = panel.property_form.labelForField(name)
            assert name_caption is not None and name_caption.geometry().right() < name.geometry().left()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #139: caption buddies, locked multi-line rows, the height grip --------


def _caption_for(panel, widget):
    """The full-width caption QLabel whose buddy is `widget`."""
    from PyQt5.QtWidgets import QLabel

    captions = [w for w in panel.property_host.findChildren(QLabel) if w.buddy() is widget]
    assert len(captions) == 1, [w.text() for w in panel.property_host.findChildren(QLabel)]
    return captions[0]


def _with_trigger_spec(monkeypatch, name: str, **changes) -> None:
    """Swap one trigger spec for a replaced copy, as the form will see it."""
    from dataclasses import replace

    from descape import trigger_fields

    original = trigger_fields.trigger_specs
    monkeypatch.setattr(
        trigger_fields,
        "trigger_specs",
        lambda version: tuple(replace(s, **changes) if s.name == name else s for s in original(version)),
    )


@pytest.mark.parametrize("field", ["description", "short_description"])
def test_a_multi_line_caption_is_its_box_s_buddy(field: str) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        widget = _row_widget(panel, field)
        assert _caption_for(panel, widget).text() == field.replace("_", " ")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_locked_multi_line_field_takes_its_caption_above_too(monkeypatch) -> None:
    """A read-only spec renders as a QLabel, not a _MultiLineEdit; its caption
    still goes on its own row with the text full width below."""
    from PyQt5.QtWidgets import QApplication, QLabel

    _with_trigger_spec(monkeypatch, "description", read_only=True)
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        widget = _row_widget(panel, "description")
        assert isinstance(widget, QLabel)
        assert panel.property_form.labelForField(widget) is None, "not a side caption"
        caption = _caption_for(panel, widget).geometry()
        form = panel.property_form.contentsRect()
        assert caption.bottom() < widget.geometry().top()
        assert (widget.geometry().left(), widget.geometry().right()) == (form.left(), form.right())
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_full_width_caption_carries_its_spec_s_tooltip(monkeypatch) -> None:
    _with_trigger_spec(monkeypatch, "description", tooltip="What the Objectives panel shows.")
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        widget = _row_widget(panel, "description")
        assert _caption_for(panel, widget).toolTip() == "What the Objectives panel shows."
        assert widget.toolTip() == "What the Objectives panel shows."
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_dragging_a_box_taller_refits_the_form_and_records_nothing() -> None:
    """The host grows by the drag's delta, no row draws over the next, and a
    resize is not a document edit: no history record, nothing dirty."""
    from PyQt5.QtWidgets import QApplication
    from test_text_edits import _drag

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        box = _row_widget(panel, "description")
        # Baseline from a fit at the shown width, the same measurement the drag's refit makes.
        panel._fit_property_height()
        host_before = panel.property_host.minimumHeight()
        box_before = box.height()

        _drag(box.grip(), 4)
        QApplication.processEvents()
        assert box.visible_lines() == box.VISIBLE_LINES + 4
        delta = box.height() - box_before
        assert delta == 4 * box.fontMetrics().lineSpacing()
        assert panel.property_host.minimumHeight() - host_before == delta
        _assert_form_rows_do_not_overlap(panel.property_form)
        assert window.trigger_edits is None
        assert not window.edit_history.can_undo
        assert not window.windowTitle().startswith("*")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_repopulating_the_form_leaves_no_empty_rows_behind() -> None:
    """takeAt() empties a QFormLayout row but keeps the row, so every populate
    used to stack another set of empty rows under the live ones."""
    from PyQt5.QtWidgets import QApplication, QFormLayout

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        form = panel.property_form
        counts = []
        for _ in range(3):
            panel._populate_property_form()
            QApplication.processEvents()
            counts.append(form.rowCount())
        assert counts[0] > 0
        assert counts == [counts[0]] * 3, f"rowCount grew across populates: {counts}"
        roles = (QFormLayout.LabelRole, QFormLayout.FieldRole, QFormLayout.SpanningRole)
        empty = [r for r in range(form.rowCount()) if all(form.itemAt(r, role) is None for role in roles)]
        assert not empty, f"empty rows left behind: {empty}"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_entering_triggers_mode_fits_the_form_to_the_shown_width() -> None:
    """The populate's fit ran while every new row still waited on Qt's queued
    show, so the host kept a 102 px minimum until something else re-fit it."""
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        QApplication.processEvents()
        panel = window.trigger_panel
        width = panel.property_area.viewport().width()
        assert width > 0
        assert panel._rows, "entering the mode no longer builds a form, so this tests nothing"
        panel.property_form.activate()
        needed = panel.property_form.heightForWidth(width)
        assert panel.property_host.minimumHeight() >= needed, (
            f"host minimum {panel.property_host.minimumHeight()} < heightForWidth({width}) {needed}"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_stored_height_is_applied_kept_across_reselection_and_a_drag_reports_once() -> None:
    from PyQt5.QtWidgets import QApplication
    from test_text_edits import _drag

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        stored = {"trigger.description": 12}
        reported = []
        panel.text_box_lines = stored.get
        panel.on_text_box_lines = lambda key, lines: reported.append((key, lines))
        # Trigger 0's form was built on entering the mode, before the injection.
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        assert _row_widget(panel, "description").visible_lines() == 12
        assert _row_widget(panel, "short_description").visible_lines() == 6, "keyed per field"

        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        box = _row_widget(panel, "description")
        assert box.visible_lines() == 12

        _drag(box.grip(), 2)
        assert reported == [("trigger.description", 14)]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_window_wires_text_box_heights_to_settings() -> None:
    """End to end through viewer.py's injection: a saved height builds the
    box, a drag saves the new one, and a double-click reset removes the key
    rather than storing the default."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from test_text_edits import _drag

    from descape import settings

    settings.set_text_box_lines("trigger.description", 9)
    settings.set_text_box_lines("messages.hints", 11)
    window = _triggers_window()
    try:
        window.mode_combo.setCurrentText("Messages")
        assert window.messages_panel.widget_for("hints").visible_lines() == 11
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        box = _row_widget(panel, "description")
        assert box.visible_lines() == 9
        _drag(box.grip(), -3)
        assert settings.get_text_box_lines("trigger.description") == 6
        QTest.mouseDClick(box.grip(), Qt.LeftButton, Qt.NoModifier, QPoint(5, 2))
        # A real double click ends with a release; QTest's leaves the button held process-wide.
        QTest.mouseRelease(box.grip(), Qt.LeftButton, Qt.NoModifier, QPoint(5, 2))
        assert settings.get_text_box_lines("trigger.description") is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    "stored", ["a\rb", "a\r\nb", "a\nb", "one line"], ids=["CR", "CRLF", "LF", "none"]
)
@pytest.mark.parametrize("where", _PROSE_WHERE)
def test_an_untouched_prose_focus_out_records_nothing(where: str, stored: str) -> None:
    """CR and CRLF display as "a\\nb", which differs from what is stored, so
    _changed()'s equality check alone would record a phantom undo step."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose(window, where, stored)
        _row_widget(panel, _PROSE_FIELD[where]).editingFinished.emit()
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    "stored,token",
    [("a\rb", "\r"), ("a\r\nb", "\r\n"), ("a\nb", "\n"), ("one line", "\n")],
    ids=["CR", "CRLF", "LF", "none"],
)
def test_an_edited_prose_field_keeps_its_own_newline_token(
    tmp_path: Path, stored: str, token: str
) -> None:
    """The behaviour the XS path deliberately does not have: XS forces every
    separator to CR, which is right for a script body and would silently
    rewrite (and widen the save diff of) a description stored with LF."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        effect_index = _plant_prose_effect(window, stored)
        widget = _row_widget(panel, "message")
        widget.setPlainText("edited\nacross\nthree lines")
        widget.editingFinished.emit()
        assert window.edit_history.is_dirty

        out = tmp_path / "prose.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded = parse_triggers(load_map_and_units(out))
        assert reloaded.triggers[0].effects[effect_index].message == token.join(
            ("edited", "across", "three lines")
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_second_prose_focus_out_is_not_a_second_edit() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose_effect(window, "a\rb")
        widget = _row_widget(panel, "message")
        widget.setPlainText("rewritten")
        widget.editingFinished.emit()
        steps = window.edit_history.cursor
        widget.editingFinished.emit()
        assert window.edit_history.cursor == steps
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_name_field_stays_one_line() -> None:
    """The other half of the set-membership decision: every *_name message is
    an identifier under 30 characters, not prose."""
    from PyQt5.QtWidgets import QLineEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose_effect(window, "Gatehouse", name="change_object_name")
        assert isinstance(_row_widget(panel, "message"), QLineEdit)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_trigger_s_own_name_stays_one_line() -> None:
    from PyQt5.QtWidgets import QLineEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_trigger_description(window, "a description")
        assert isinstance(_row_widget(panel, "name"), QLineEdit)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_locked_prose_field_renders_its_lines() -> None:
    from PyQt5.QtWidgets import QLabel

    from descape.trigger_fields import PROSE, STR, FieldSpec

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose_effect(window, "a\r\nb")
        spec = FieldSpec("message", STR, sentinel=None, read_only=True, multiline=PROSE)
        label = panel._build_widget(spec, [("effect", 0)], panel._read(panel._current_entry()[2], spec.attribute))
        assert isinstance(label, QLabel)
        assert label.text() == "a\nb"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- retyping an existing condition or effect (GH #37) -----------------------


def _select_first_effect(panel) -> int:
    """Select the first effect the fixture carries, and return its trigger
    index. Asserts rather than skips, for _select_first_condition's reason."""
    for i in range(panel.tree.topLevelItemCount()):
        panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
        effects = _group(panel, "effect")
        if effects.childCount():
            panel.entry_tree.setCurrentItem(effects.child(0))
            return i
    raise AssertionError("the fixture carries no effects, so retype tests cannot run")


def _effects_of(window, trigger_index: int):
    manager = window.trigger_panel._manager()
    return manager.triggers[trigger_index].effects


def test_the_retype_picker_offers_one_kind_and_opens_on_the_current_type() -> None:
    """Cross-kind retyping is not offered -- conditions and effects are
    separate lists -- and the picker opens where the user already is."""
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        trigger_index = _select_first_effect(panel)
        current_type = _effects_of(window, trigger_index)[0].effect_type

        panel._request_entry_op("retype")
        assert panel.detail_stack.currentIndex() == 1
        assert panel.picker_tree.topLevelItemCount() == 1
        assert "Effects" in panel.picker_tree.topLevelItem(0).text(0)
        assert panel.picker_add_button.text() == "Change"

        picked = panel.picker_tree.currentItem()
        assert picked is not None, "the retype picker opened with nothing selected"
        assert picked.data(0, Qt.UserRole) == ("effect", current_type)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_opening_and_cancelling_the_retype_picker_records_nothing(tmp_path: Path) -> None:
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_first_effect(panel)
        panel._request_entry_op("retype")
        _pick(panel, "effect", "send chat")
        panel.picker_filter.setText("chat")
        panel._close_picker()

        assert panel.detail_stack.currentIndex() == 0
        assert panel._picker_entry_ref is None, "the entry latch outlived the picker"
        assert window.trigger_edits is None, "opening the picker must not build an edit model"
        assert not window.edit_history.is_dirty

        out = tmp_path / "cancelled.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_picking_the_type_the_entry_already_has_records_nothing() -> None:
    """commit_trigger_edit() pushes unconditionally, so _accept_pick() is the
    only place a no-op retype can be stopped."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_first_effect(panel)
        panel._request_entry_op("retype")
        # The picker already opened on the current type, so accept as-is.
        panel._accept_pick()

        assert panel.detail_stack.currentIndex() == 0
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_retype_targets_the_entry_latched_at_open() -> None:
    """The detail tree stays live behind the picker in Add mode; in Change mode
    a selection change in between must not move the target."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        trigger_index = _select_first_effect(panel)
        effects = _effects_of(window, trigger_index)
        if len(effects) < 2:
            for i in range(panel.tree.topLevelItemCount()):
                panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
                if _group(panel, "effect").childCount() >= 2:
                    trigger_index = i
                    break
            effects = _effects_of(window, trigger_index)
        assert len(effects) >= 2, "the fixture has no trigger with two effects"
        panel.entry_tree.setCurrentItem(_group(panel, "effect").child(0))

        before = [e.effect_type for e in effects]
        panel._request_entry_op("retype")
        # Move the detail selection to the *second* effect while the picker is
        # up. The tree is hidden there, but a programmatic selection still
        # fires _on_entry_selected, and the latch is what has to survive it --
        # re-reading the selection at accept time would retype the wrong entry.
        panel.entry_tree.setCurrentItem(_group(panel, "effect").child(1))
        _pick(panel, "effect", "send chat")
        panel._accept_pick()

        after = [e.effect_type for e in window.trigger_edits.manager().triggers[trigger_index].effects]
        assert after[0] != before[0], "the latched entry was not retyped"
        assert after[1:] == before[1:], "a retype moved off the entry latched at open"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_retype_keeps_the_entry_in_place_and_re_anchors_its_uuid() -> None:
    """Build-then-replace, not append: the index, the list length and the
    display-order array all have to come out unchanged, and UuidList.__setitem__
    is what re-anchors the fresh object to the document."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        trigger_index = _select_first_effect(panel)
        effects = _effects_of(window, trigger_index)
        before_len = len(effects)
        before_order = list(
            panel._manager().triggers[trigger_index].effect_order
        )

        window.entry_structural_edit("retype", trigger_index, "effect", 0, 3)

        trigger = window.trigger_edits.manager().triggers[trigger_index]
        assert len(trigger.effects) == before_len, "a retype changed the list length"
        assert trigger.effects[0].effect_type == 3
        assert list(trigger.effect_order) == before_order, "a retype rewrote the display order"
        assert trigger.effects[0]._uuid == trigger._uuid, "the fresh entry was never re-anchored"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_retype_carries_the_shared_fields_and_reports_the_rest() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        # Trigger 0's effect is display_instructions, whose message/source_player
        # send_chat also lists, and whose display_time it does not.
        trigger_index = _select_first_effect(panel)
        effect = _effects_of(window, trigger_index)[0]
        effect.message = "carried"
        effect.source_player = 2
        effect.display_time = 15

        window.entry_structural_edit("retype", trigger_index, "effect", 0, 3)

        fresh = window.trigger_edits.manager().triggers[trigger_index].effects[0]
        assert fresh.effect_type == 3
        assert fresh.message == "carried"
        assert fresh.source_player == 2
        assert "display time" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- multi-select and clipboard (GH #27/#28) ----------------------------------


def _ctrl_select(panel, *indices) -> None:
    """Qt's Ctrl+click, minus the mouse: current first, then add to the set."""
    panel.select_triggers(list(indices))


def test_the_trigger_and_entry_trees_are_multi_select_and_the_picker_is_not() -> None:
    """The picker resolves to exactly one vocabulary row; both content trees
    take a set (#27, #60)."""
    from PyQt5.QtWidgets import QAbstractItemView

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        for tree in (panel.tree, panel.entry_tree):
            assert tree.selectionMode() == QAbstractItemView.ExtendedSelection
        assert panel.picker_tree.selectionMode() == QAbstractItemView.SingleSelection
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selected_trigger_indices_follow_display_order_under_either_sort() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        window.trigger_structural_edit("move_up", [3])  # display order [0, 1, 3, 2]
        _ctrl_select(panel, 2, 0, 3)
        assert panel.selected_trigger_indices() == [0, 3, 2]
        panel.sort_combo.setCurrentIndex(1)  # File order
        _ctrl_select(panel, 2, 0, 3)
        assert panel.selected_trigger_indices() == [0, 3, 2]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_filtered_out_row_is_never_in_the_selection() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 0, 1)
        panel.filter_edit.setText("setup")  # hides "Fixture: armour split"
        assert panel.selected_trigger_indices() == [0]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_multi_selection_shows_a_count_and_hides_the_entry_tree_contents() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 0, 2)
        assert panel.detail_stack.currentIndex() == panel._DETAIL_MULTI
        assert panel.multi_label.text() == "2 triggers selected"
        assert panel.entry_tree.topLevelItemCount() == 0
        for button in (panel.entry_new_button, panel.entry_copy_button, panel.entry_delete_button):
            assert not button.isEnabled()

        panel.select_trigger(1)
        assert panel.detail_stack.currentIndex() == panel._DETAIL_FORM
        assert panel.entry_tree.topLevelItemCount() == 3, "Trigger, Conditions, Effects"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_deselecting_down_to_one_restores_the_form_without_a_current_change() -> None:
    """currentItemChanged does not fire when the current row stays put, so the
    selection handler must bring the entry tree back itself."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 0, 2)
        panel._item_for_index[2].setSelected(False)
        assert panel.current_trigger_index() == 0
        assert panel.detail_stack.currentIndex() == panel._DETAIL_FORM
        assert panel.entry_tree.topLevelItemCount() == 3
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_triggers_keeps_the_whole_set() -> None:
    """Trap 7: setCurrentItem() clears the selection, so building the set and
    then setting the current item would leave one row selected."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_triggers([3, 1])
        assert panel.current_trigger_index() == 3
        assert panel.selected_trigger_indices() == [1, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_same_document_rebuild_keeps_the_selected_set() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1, 3)
        panel.show_scenario(window.scenario)
        assert panel.current_trigger_index() == 1
        assert panel.selected_trigger_indices() == [1, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_multi_select_walk_saves_byte_identically(tmp_path: Path) -> None:
    """Slice 0 is read-only: selecting sets, changing them and clearing them
    must not build a model or dirty a byte."""
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        for selection in ([0, 1], [0, 1, 2, 3], [3, 1], [2]):
            _ctrl_select(panel, *selection)
        panel.sort_combo.setCurrentIndex(1)
        _ctrl_select(panel, 0, 3)
        panel.sort_combo.setCurrentIndex(0)
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
        out = tmp_path / "walked.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_buttons_follow_the_block_bounds() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 0, 2)
        assert not panel.trigger_move_up_button.isEnabled(), "the leading member is at slot 0"
        assert panel.trigger_move_down_button.isEnabled()
        _ctrl_select(panel, 1, 3)
        assert panel.trigger_move_up_button.isEnabled()
        assert not panel.trigger_move_down_button.isEnabled(), "the trailing member is last"
        _ctrl_select(panel, 1, 2)
        assert panel.trigger_move_up_button.isEnabled() and panel.trigger_move_down_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_buttons_act_on_the_whole_selection() -> None:
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1, 2)
        panel.trigger_move_down_button.click()
        assert panel._display_slots == [0, 3, 1, 2]
        assert panel.selected_trigger_indices() == [1, 2], "the moved block stays selected"
        assert [panel.tree.topLevelItem(i).data(0, Qt.UserRole) for i in range(4)] == [0, 3, 1, 2]

        panel.trigger_delete_button.click()
        names = [t.name for t in window.trigger_edits.manager().triggers]
        assert names == ["Fixture: setup", "Fixture: variable"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copying_two_triggers_duplicates_each_in_place_with_links_to_the_originals() -> None:
    """Copy is N independent duplicates: the copy of "references" still
    activates the original "setup". Copy + Paste is the linked-block verb."""
    from descape.trigger_clipboard import get_trigger_referencing_ce

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        window.trigger_structural_edit("move_up", [3])  # display [0, 1, 3, 2]
        window.trigger_structural_edit("copy", [0, 2])
        manager = window.trigger_edits.manager()
        names = [manager.triggers[i].name for i in manager.trigger_display_order]
        assert names == [
            "Fixture: setup", "Fixture: setup (copy)", "Fixture: armour split",
            "Fixture: variable", "Fixture: references", "Fixture: references (copy)",
        ]
        copies = [manager.triggers[i] for i in panel.selected_trigger_indices()]
        assert [t.name for t in copies] == ["Fixture: setup (copy)", "Fixture: references (copy)"]
        setup = next(i for i, t in enumerate(manager.triggers) if t.name == "Fixture: setup")
        assert get_trigger_referencing_ce(copies[1])[0].trigger_id == setup
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_then_paste_lands_the_block_below_the_current_trigger() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert not panel.trigger_paste_button.isEnabled(), "nothing to paste yet"
        _ctrl_select(panel, 0, 2)
        window.copy_triggers()
        assert panel.trigger_paste_button.isEnabled()
        assert window.trigger_edits is None or not window.trigger_edits.has_edits, "copying records nothing"

        panel.select_trigger(1)
        panel.trigger_paste_button.click()
        manager = window.trigger_edits.manager()
        assert list(manager.trigger_display_order) == [0, 1, 4, 5, 2, 3]
        assert panel.selected_trigger_indices() == [4, 5], "the pasted block is selected"
        # A linked block: the pasted "references" activates the pasted "setup".
        refs = [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[5])]
        assert refs == [4, 1]

        window.undo()
        assert not window.edit_history.is_dirty, "the paste was one undo step"
        assert len(window.trigger_edits.manager().triggers) == 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def _row_buttons(layout) -> list:
    return [layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget() is not None]


def test_the_copy_button_fills_the_clipboard_without_adding_triggers() -> None:
    """GH #27/#28: with Paste beside it, "Copy" copies; Paste then enables."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        copy = next(b for b in _row_buttons(panel.trigger_buttons) if b.text() == "Copy")
        _ctrl_select(panel, 0, 2)
        assert copy.isEnabled()
        assert not panel.trigger_paste_button.isEnabled(), "nothing to paste yet"
        copy.click()
        assert len(window.trigger_panel._manager().triggers) == 4, "copying adds nothing"
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
        assert [t.name for t in window._trigger_clipboard.triggers] == ["Fixture: setup", "Fixture: references"]
        assert panel.trigger_paste_button.isEnabled()
        panel.tree.clearSelection()
        assert not copy.isEnabled(), "nothing selected to copy"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_duplicate_buttons_keep_their_names_and_still_duplicate() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert panel.trigger_copy_button.text() == "Duplicate"
        assert panel.entry_copy_button.text() == "Duplicate"
        assert [b.text() for b in _row_buttons(panel.trigger_buttons)] == ["New", "Copy", "Paste", "Delete"]
        assert panel.trigger_copy_button in _row_buttons(panel.reorder_buttons)
        _ctrl_select(panel, 0)
        panel.trigger_copy_button.click()
        assert len(window.trigger_edits.manager().triggers) == 5
        assert window._trigger_clipboard is None, "Duplicate leaves the clipboard alone"
    finally:
        window.edit_history.mark_saved()
        window.close()


def trigger_clipboard_refs(trigger):
    from descape.trigger_clipboard import get_trigger_referencing_ce

    return get_trigger_referencing_ce(trigger)


def test_paste_with_no_current_trigger_appends(tmp_path: Path) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 3)
        window.copy_triggers()
        window.trigger_structural_edit("paste", [])
        assert list(window.trigger_edits.manager().trigger_display_order) == [0, 1, 2, 3, 4]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_clipboard_from_another_scenario_version_is_refused_out_loud(monkeypatch) -> None:
    """GH #3: Paste stays enabled, and Ctrl+V or the button logs why it
    refused, changing nothing. QMessageBox.warning is patched: unpatched it
    blocks forever offscreen."""
    import dataclasses

    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    from descape import viewer as viewer_module

    warnings = []
    monkeypatch.setattr(viewer_module.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _triggers_window()
    try:
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        panel = window.trigger_panel
        _ctrl_select(panel, 0)
        window.copy_triggers()
        window._trigger_clipboard = dataclasses.replace(window._trigger_clipboard, scenario_version="1.56")
        window._sync_trigger_clipboard_state()
        expected = "Copied from a 1.56 scenario; this one is 1.58."
        assert panel.trigger_paste_button.isEnabled()
        assert window.paste_action.isEnabled()
        assert panel.trigger_paste_button.toolTip() == expected
        assert window.paste_action.toolTip() == expected
        window.status_log.clear()

        QTest.keyClick(window, Qt.Key_V, Qt.ControlModifier)
        QApplication.processEvents()
        assert f"Cannot paste these triggers: {expected}" in window.status_log.toPlainText()
        assert len(panel._manager().triggers) == 4

        window.status_log.clear()
        panel.trigger_paste_button.click()
        assert f"Cannot paste these triggers: {expected}" in window.status_log.toPlainText()
        assert len(panel._manager().triggers) == 4
        assert warnings == []
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
        assert not window.edit_history.is_dirty, "no undo record was pushed"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


def test_a_paste_into_another_file_reports_and_undoes_in_one_step() -> None:
    """GH #3 through the window: copy in the fixture, File > New Map, paste.
    Both outside links are cleared, variable 0 takes the source's name, and one
    undo removes the triggers and the name together."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 2, 3)
        window.copy_triggers()
        window.edit_history.mark_saved()
        window.new_map()
        window.mode_combo.setCurrentText("Triggers")
        assert panel.trigger_paste_button.isEnabled(), "the clipboard outlives the document"

        window.trigger_structural_edit("paste", [])
        manager = window.trigger_edits.manager()
        assert [t.name for t in manager.triggers] == ["Fixture: references (copy)", "Fixture: variable (copy)"]
        assert [(v.variable_id, v.name) for v in manager.variables] == [(0, "fixture_var")]
        # The 5 unit references are the fixture's own, on trigger 3.
        assert "Pasted 2 triggers from another scenario: cleared 2 trigger links and 5 unit references, named 1 variable." in (
            window.status_log.toPlainText()
        )

        window.undo()
        manager = window.trigger_edits.manager()
        assert (len(manager.triggers), len(manager.variables)) == (0, 0)
        assert not window.edit_history.is_dirty, "the paste was one undo step"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_paste_after_reopening_the_same_file_counts_as_another_scenario() -> None:
    """GH #3's reopen check: a reload is a new document, so the links a copy
    carried are cleared even though the path is the same."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 2, 3)
        window.copy_triggers()
        window.edit_history.mark_saved()
        window.load_scenario(TRIGGER_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        window.status_log.clear()

        window.trigger_structural_edit("paste", [])
        assert [t.name for t in window.trigger_edits.manager().triggers][-2:] == [
            "Fixture: references (copy)",
            "Fixture: variable (copy)",
        ]
        # Variable 0 already carries the same name here, so none is renamed.
        assert "Pasted 2 triggers from another scenario: cleared 2 trigger links and 5 unit references." in (
            window.status_log.toPlainText()
        ), window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_ctrl_a_in_the_filter_box_selects_its_text_not_the_triggers() -> None:
    """GH #3: the line edit claims Ctrl+A before the window's Select All does."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        # Shortcut dispatch needs an active window, as in test_keybinds.py.
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        assert window.isActiveWindow()

        panel = window.trigger_panel
        panel.select_trigger(1)
        panel.filter_edit.setText("Fixture")  # matches every shipped trigger
        panel.filter_edit.setFocus()
        QApplication.processEvents()
        assert panel.filter_edit.hasFocus()
        assert panel.selected_trigger_indices() == [1]

        QTest.keyClick(panel.filter_edit, Qt.Key_A, Qt.ControlModifier)
        QApplication.processEvents()
        assert panel.filter_edit.selectedText() == "Fixture"
        assert panel.selected_trigger_indices() == [1]
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


def test_select_all_in_triggers_mode_takes_every_visible_trigger() -> None:
    """Ctrl+A dispatches on mode like Ctrl+C: every trigger in Triggers mode,
    collapsed sections included, filtered rows excluded; the whole map elsewhere."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 2, "--- Second ---")  # sections: (before) 0, 1 | 2: 3
        panel._item_for_index[2].setExpanded(False)
        window.select_all_action.trigger()
        assert panel.selected_trigger_indices() == [0, 1, 2, 3]
        assert window.deselect_action.isEnabled()
        window.deselect_action.trigger()
        assert panel.selected_trigger_indices() == []
        # The focused tree's own Ctrl+A lands here, not on the window's action.
        panel.tree.selectAll()
        assert panel.selected_trigger_indices() == [0, 1, 2, 3]

        panel.filter_edit.setText("variable")
        window.select_all_action.trigger()
        # The divider stays on screen for its matching member, so it counts.
        assert panel.selected_trigger_indices() == [2, 3]

        window.mode_combo.setCurrentText("Terrain")
        window.select_all_action.trigger()
        mm = window.scenario.map_manager
        assert window._region == (0, 0, mm.map_width, mm.map_height)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_programmatic_selection_replaces_even_with_ctrl_held() -> None:
    """Ctrl+V arrives with Ctrl down, and the post-edit select_triggers() must
    still replace the selection rather than toggle it."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        QTest.keyClick(window, Qt.Key_A, Qt.ControlModifier)  # leaves Ctrl held
        panel.select_triggers([3, 1])
        assert panel.selected_trigger_indices() == [1, 3]
        panel.select_trigger(2)
        assert panel.selected_trigger_indices() == [2]
    finally:
        for key in (Qt.Key_Control,):
            QTest.keyRelease(window, key)
        window.edit_history.mark_saved()
        window.close()


def _rename(panel, index: int, name: str) -> None:
    panel.select_trigger(index)
    widget = _row_widget(panel, "name")
    widget.setText(name)
    widget.editingFinished.emit()


def _action(menu, text: str):
    return next(a for a in menu.actions() if a.text() == text)


def test_select_section_selects_the_divider_and_its_members() -> None:
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        flat_menu = panel.trigger_context_menu(panel._item_for_index[1])
        assert not _action(flat_menu, "Select Section").isEnabled(), "no sections while flat"

        _rename(panel, 2, "--- Second ---")  # sections: (before) 0, 1 | 2: 3
        assert panel._grouped
        _action(panel.trigger_context_menu(panel._item_for_index[3]), "Select Section").trigger()
        assert panel.selected_trigger_indices() == [2, 3]

        synthetic = panel.tree.topLevelItem(0)
        assert synthetic.data(0, Qt.UserRole) is None
        _action(panel.trigger_context_menu(synthetic), "Select Section").trigger()
        assert panel.selected_trigger_indices() == [0, 1], "the synthetic header itself is never selected"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_tag_spans_sections_and_skips_hidden_rows() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 0, "[a] one")
        _rename(panel, 3, "[a] two")
        _rename(panel, 2, "--- Second ---")
        assert not _action(panel.trigger_context_menu(panel._item_for_index[1]), "Select Tag").isEnabled()

        _action(panel.trigger_context_menu(panel._item_for_index[0]), "Select Tag").trigger()
        assert panel.selected_trigger_indices() == [0, 3]

        panel.filter_edit.setText("two")
        _action(panel.trigger_context_menu(panel._item_for_index[3]), "Select Tag").trigger()
        assert panel.selected_trigger_indices() == [3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_edit_copy_and_paste_dispatch_on_mode() -> None:
    """One QAction pair keeps Ctrl+C / Ctrl+V; in Triggers mode it retitles
    and acts on triggers, tracking the selection as it changes."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert window.copy_action.text() == "&Copy Triggers"
        assert window.paste_action.text() == "&Paste Triggers"
        assert not window.paste_action.isEnabled(), "nothing copied yet"

        panel.tree.clearSelection()
        assert not window.copy_action.isEnabled()
        _ctrl_select(panel, 1, 3)
        assert window.copy_action.isEnabled()

        window.copy_action.trigger()
        assert window._trigger_clipboard is not None and len(window._trigger_clipboard.triggers) == 2
        assert window.paste_action.isEnabled()

        panel.select_trigger(0)
        window.paste_action.trigger()
        assert list(window.trigger_edits.manager().trigger_display_order) == [0, 4, 5, 1, 2, 3]

        window.mode_combo.setCurrentText("Terrain")
        assert window.copy_action.text() == "&Copy Region"
        assert window.paste_action.text() == "&Paste Region"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- tag management: Rename tag… and its Remove tag… menu --------------------

_TAGGED_NAMES = ["[D1] setup", "[d1]armour split", "[P1][D1] references", "[D1]: variable"]


def _tagged_window(tmp_path: Path, names: list[str] = _TAGGED_NAMES):
    """A window on the trigger fixture with tagged names saved into it: D1 x2,
    a case variant d1, and P1 with D1 only as its chained second tag."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = TriggerEditModel(loaded)
    for index, name in enumerate(names):
        model.manager().triggers[index].name = name
        model.mark_dirty(index)
    base = tmp_path / "tagged.aoe2scenario"
    write_scenario(loaded, base, triggers=model)
    window = _window()
    window.load_scenario(base)
    window.mode_combo.setCurrentText("Triggers")
    return window


def _names(window) -> list[str]:
    return [t.name for t in window.trigger_panel._manager().triggers]


def _visible_ids(panel) -> list[int]:
    return sorted(i for i, item in panel._item_for_index.items() if not item.isHidden())


def _answer_dialogs(monkeypatch, text=None, answer=None) -> dict:
    """Stub the panel's two dialogs; record what each was asked."""
    from descape import trigger_panel

    asked = {"getText": [], "question": []}

    def get_text(parent, title, label, mode, default):
        asked["getText"].append((title, label, default))
        return (text, True) if text is not None else ("", False)

    def question(parent, title, body, *rest):
        asked["question"].append(body)
        return answer

    monkeypatch.setattr(trigger_panel.QInputDialog, "getText", staticmethod(get_text))
    monkeypatch.setattr(trigger_panel.QMessageBox, "question", staticmethod(question))
    return asked


def test_rename_tag_is_offered_only_on_a_chosen_tag_of_a_writable_file(tmp_path: Path) -> None:
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        assert panel.current_tag() is None
        assert not panel.tag_rename_button.isEnabled()
        assert not panel.tag_remove_action.isEnabled()
        assert panel.tag_rename_button.toolTip() == "Pick a tag to rename"

        panel.set_tag_filter("D1")
        assert panel.tag_rename_button.isEnabled()
        assert panel.tag_remove_action.isEnabled()
        # Not gated on the text filter or the sort mode.
        panel.filter_edit.setText("setup")
        panel.sort_combo.setCurrentIndex(1)
        assert panel.tag_rename_button.isEnabled()

        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        assert panel.current_tag() == "D1"
        assert not panel.tag_rename_button.isEnabled()
        assert not panel.tag_remove_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_picked_tag_disables_move_up_and_down_and_says_why(tmp_path: Path) -> None:
    """GH #1: the tag facet hides rows like the text filter, so a swap could
    land on a hidden neighbour; both buttons disable with the tag named as why."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("P1")
        assert _visible_ids(panel) == [2], "fixture assumption: one P1 trigger, away from both ends"
        panel.select_trigger(2)
        for button in (panel.trigger_move_up_button, panel.trigger_move_down_button):
            assert not button.isEnabled()
            assert "tag" in button.toolTip(), button.toolTip()

        panel.set_tag_filter(None)
        panel.select_trigger(2)
        assert panel.trigger_move_up_button.isEnabled() and panel.trigger_move_down_button.isEnabled()
        assert "tag" not in panel.trigger_move_up_button.toolTip()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_the_facets_tag_keeps_the_facet_on_it(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path)
    asked = _answer_dialogs(monkeypatch, text="  Intro ")
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("D1")
        assert _visible_ids(panel) == [0, 2, 3], "D1 chained second on [P1][D1] counts too"
        panel.tag_rename_button.click()
        assert asked["getText"] == [('Rename tag "D1"', "3 triggers carry this tag. New tag:", "D1")]
        assert asked["question"] == []
        assert _names(window) == ["[Intro] setup", "[d1]armour split", "[P1][Intro] references", "[Intro]: variable"]
        assert window.edit_history.records[-1].touched == [0, 2, 3]
        assert panel.current_tag() == "Intro"
        assert _visible_ids(panel) == [0, 2, 3]
        assert panel.tag_combo.findData("D1") < 0
        # Undo drops the facet to "All tags": the renamed tag is gone again.
        window.undo()
        assert _names(window) == _TAGGED_NAMES
        assert panel.current_tag() is None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_cancelled_rename_calls_nothing(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path)
    _answer_dialogs(monkeypatch, text=None)
    try:
        window.trigger_panel.set_tag_filter("D1")
        window.trigger_panel.request_tag_rename()
        assert window.edit_history.records == []
    finally:
        window.close()


def test_declining_a_merge_calls_nothing(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    calls = []
    window = _tagged_window(tmp_path)
    monkeypatch.setattr(window.trigger_panel, "_on_tag_rename", lambda *args: calls.append(args))
    asked = _answer_dialogs(monkeypatch, text="D1", answer=QMessageBox.No)
    try:
        window.trigger_panel.set_tag_filter("d1")
        window.trigger_panel.request_tag_rename()
        assert asked["question"] == [
            'Merge tag "d1" (1 trigger) into existing tag "D1" (3 triggers)? One undo reverses it.'
        ]
        assert calls == []
        assert window.edit_history.records == []
    finally:
        window.close()


def test_accepting_a_merge_leaves_one_facet_holding_both(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    window = _tagged_window(tmp_path)
    _answer_dialogs(monkeypatch, text="D1", answer=QMessageBox.Yes)
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("d1")
        panel.request_tag_rename()
        assert panel.tag_combo.findData("d1") < 0
        assert panel.current_tag() == "D1"
        assert _visible_ids(panel) == [0, 1, 2, 3]
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_remove_tag_strips_without_asking_in_one_undo_step(tmp_path: Path, monkeypatch) -> None:
    """GH #101: no confirmation, since one undo reverses it."""
    window = _tagged_window(tmp_path)
    asked = _answer_dialogs(monkeypatch)
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("D1")
        panel.tag_remove_action.trigger()
        assert asked == {"getText": [], "question": []}
        assert _names(window) == ["setup", "[d1]armour split", "[P1] references", ": variable"]
        assert window.edit_history.records[-1].touched == [0, 2, 3]
        assert panel.current_tag() is None
        assert panel.tag_combo.findData("D1") < 0
        assert len(window.edit_history.records) == 1
        window.undo()
        assert _names(window) == _TAGGED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()


class _Asked(list):
    """_answer_item's record, plus the item each prompt pre-selected."""

    defaults: list


def _answer_item(monkeypatch, text=None) -> _Asked:
    """Stub the panel's QInputDialog.getItem; record (title, label, items, editable)."""
    from descape import trigger_panel

    asked = _Asked()
    asked.defaults = []

    def get_item(parent, title, label, items, current=0, editable=True):
        asked.append((title, label, list(items), editable))
        asked.defaults.append(list(items)[current] if items else None)
        return (text, True) if text is not None else ("", False)

    monkeypatch.setattr(trigger_panel.QInputDialog, "getItem", staticmethod(get_item))
    return asked


def test_add_tag_tags_the_selected_triggers_lacking_it_in_one_step(monkeypatch) -> None:
    """GH #101: existing tags are offered, a trigger already carrying the tag
    is left alone, and one undo reverses the lot."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.clearSelection()
        assert not panel.tag_add_button.isEnabled(), "nothing selected"
        _ctrl_select(panel, 0, 2)
        assert panel.tag_add_button.isEnabled()
        asked = _answer_item(monkeypatch, text="  Intro ")
        panel.tag_add_button.click()
        assert asked == [("Add tag", "Add a tag to 2 selected triggers:", [], True)]
        assert _names(window) == [
            "[Intro] Fixture: setup", "Fixture: armour split", "[Intro] Fixture: references", "Fixture: variable",
        ]
        assert panel.tag_combo.findData("Intro") >= 0
        assert panel.selected_trigger_indices() == [0, 2], "the selection survives"
        assert len(window.edit_history.records) == 1
        window.undo()
        assert _names(window) == ["Fixture: setup", "Fixture: armour split", "Fixture: references", "Fixture: variable"]
        window.redo()

        _ctrl_select(panel, 0, 1)
        asked = _answer_item(monkeypatch, text="Intro")
        panel.tag_add_button.click()
        assert asked[0][2] == ["Intro"], "existing tags are offered"
        assert _names(window)[:2] == ["[Intro] Fixture: setup", "[Intro] Fixture: armour split"]
        assert len(window.edit_history.records) == 2
        window.undo()
        assert _names(window)[:2] == ["[Intro] Fixture: setup", "Fixture: armour split"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_add_tag_chains_onto_another_tag_and_refuses_a_closer(tmp_path: Path, monkeypatch) -> None:
    """Add tag prepends, so adding to "[d1]..." chains in front
    ("[P1][D1] ..." is the corpus's own shape); a tag holding "]" would not
    read back, so it is refused before anything is recorded."""
    from descape import viewer as viewer_module

    warnings = []
    monkeypatch.setattr(viewer_module.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1)
        _answer_item(monkeypatch, text="a]b")
        panel.tag_add_button.click()
        assert len(warnings) == 1 and '"a]b"' in warnings[0]
        assert window.edit_history.records == []

        _answer_item(monkeypatch, text="New")
        panel.tag_add_button.click()
        assert _names(window)[1] == "[New] [d1]armour split"
        _answer_item(monkeypatch)  # cancelled
        panel.tag_add_button.click()
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_add_tag_skips_a_trigger_carrying_it_anywhere_in_the_chain(tmp_path: Path, monkeypatch) -> None:
    """"[P1][D1] references" already carries D1 as its second tag, so Add tag
    D1 leaves it alone; "[d1]..." (case differs) still gets one, in one step."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 2)
        _answer_item(monkeypatch, text="D1")
        panel.tag_add_button.click()
        assert _names(window) == _TAGGED_NAMES
        assert window.edit_history.records == []

        _ctrl_select(panel, 1, 2)
        _answer_item(monkeypatch, text="D1")
        panel.tag_add_button.click()
        assert _names(window) == ["[D1] setup", "[D1] [d1]armour split", "[P1][D1] references", "[D1]: variable"]
        assert len(window.edit_history.records) == 1
        record = window.edit_history.records[-1]
        assert record.touched == [1] and record.label == 'Add tag "D1" (1 trigger)'
        window.undo()
        assert _names(window) == _TAGGED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_remove_tag_on_a_selection_strips_only_the_selected_carriers(tmp_path: Path, monkeypatch) -> None:
    """The facet's Remove tag stays the every-trigger form; this one asks
    which of the selection's tags to drop, and does not ask when there is one."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 0, 2)
        assert panel.tag_remove_selected_button.isEnabled()
        asked = _answer_item(monkeypatch, text="D1")
        panel.tag_remove_selected_button.click()
        assert asked == [("Remove tag", "Remove a tag from 2 selected triggers:", ["D1", "P1"], False)]
        assert asked.defaults == ["D1"], "the first selected row's leading tag"
        assert _names(window) == ["setup", "[d1]armour split", "[P1] references", "[D1]: variable"]
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[-1].touched == [0, 2]
        window.undo()
        assert _names(window) == _TAGGED_NAMES

        _ctrl_select(panel, 2)
        asked = _answer_item(monkeypatch, text="D1")
        panel.tag_remove_selected_button.click()
        assert asked == [("Remove tag", "Remove a tag from 1 selected trigger:", ["D1", "P1"], False)]
        assert asked.defaults == ["P1"], "pre-selects the leading tag, not the sorted-first chained one"
        assert _names(window)[2] == "[P1] references", "the chained second tag, stripped in place"
        window.undo()
        assert _names(window) == _TAGGED_NAMES

        _ctrl_select(panel, 0)
        asked = _answer_item(monkeypatch, text="unused")
        panel.tag_remove_selected_button.click()
        assert asked == [], "one tag in the selection: nothing to choose"
        assert _names(window)[0] == "setup"
        window.undo()
        assert _names(window) == _TAGGED_NAMES
        _ctrl_select(panel, 0)
        window.trigger_structural_edit("new", [])
        _ctrl_select(panel, 4)
        assert not panel.tag_remove_selected_button.isEnabled(), "an untagged selection has nothing to remove"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_tag_made_by_renaming_a_trigger_joins_the_dropdown_at_once(tmp_path: Path) -> None:
    """GH #101: no mode switch needed for the facet to offer it."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _rename(panel, 1, "[Fresh] armour split")
        assert panel.tag_combo.findData("Fresh") >= 0
        assert panel.tag_combo.findData("d1") < 0, "its old tag had no other carrier"
        panel.set_tag_filter("Fresh")
        assert _visible_ids(panel) == [1]
        window.undo()
        assert panel.tag_combo.findData("Fresh") < 0
        assert panel.tag_combo.findData("d1") >= 0
    finally:
        window.edit_history.mark_saved()
        window.close()


_CHAINED_NAMES = ["[New] setup", "[Old] [New] x", "[Old] solo", "plain"]


def _combo_tags(panel) -> list[str]:
    return [panel.tag_combo.itemData(i) for i in range(1, panel.tag_combo.count())]


def test_a_tag_anywhere_in_the_chain_is_listed_counted_and_filtered(tmp_path: Path) -> None:
    """The TODO's own "[Old] [New] x": New is listed, counted and matched,
    not only the leading Old. New leads nowhere, so only the chain finds it."""
    window = _tagged_window(tmp_path, ["[Old] [New] x", "[Old] solo", "plain", "plain"])
    try:
        panel = window.trigger_panel
        assert _combo_tags(panel) == ["New", "Old"]
        assert panel._tag_count("New") == 1
        assert panel._tag_count("Old") == 2
        panel.set_tag_filter("New")
        assert _visible_ids(panel) == [0]
        panel.set_tag_filter("Old")
        assert _visible_ids(panel) == [0, 1]
        _rename(panel, 3, "--- Sec ---")
        assert panel._grouped, "fixture assumption"
        panel.set_tag_filter("New")
        assert _visible_ids(panel) == [0], "the grouped tree filters the same way"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_remove_tag_strips_a_chained_tag_in_place_in_one_undo_step(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path, _CHAINED_NAMES)
    asked = _answer_dialogs(monkeypatch)
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("New")
        panel.tag_remove_action.trigger()
        assert asked == {"getText": [], "question": []}
        assert _names(window) == ["setup", "[Old] x", "[Old] solo", "plain"]
        record = window.edit_history.records[-1]
        assert len(window.edit_history.records) == 1
        assert record.touched == [0, 1] and record.label == 'Remove tag "New" (2 triggers)'
        assert panel.tag_combo.findData("New") < 0
        window.undo()
        assert _names(window) == _CHAINED_NAMES
        assert panel.tag_combo.findData("New") >= 0
        window.redo()
        assert _names(window) == ["setup", "[Old] x", "[Old] solo", "plain"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rename_tag_retags_a_chained_tag_in_place_in_one_undo_step(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path, _CHAINED_NAMES)
    asked = _answer_dialogs(monkeypatch, text="Newer")
    try:
        panel = window.trigger_panel
        panel.set_tag_filter("New")
        panel.tag_rename_button.click()
        assert asked["getText"] == [('Rename tag "New"', "2 triggers carry this tag. New tag:", "New")]
        assert _names(window) == ["[Newer] setup", "[Old] [Newer] x", "[Old] solo", "plain"]
        record = window.edit_history.records[-1]
        assert len(window.edit_history.records) == 1 and record.touched == [0, 1]
        assert panel.current_tag() == "Newer"
        assert _visible_ids(panel) == [0, 1]
        window.undo()
        assert _names(window) == _CHAINED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_closer_in_a_chained_tags_new_name_is_refused(tmp_path: Path, monkeypatch) -> None:
    from descape import viewer as viewer_module

    warnings = []
    monkeypatch.setattr(viewer_module.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _tagged_window(tmp_path, ["[Old] [New] x", "plain", "plain", "plain"])
    try:
        window.rename_trigger_tag("New", "a]b")
        assert window.edit_history.records == []
        assert _names(window)[0] == "[Old] [New] x"
    finally:
        window.close()
    assert warnings == ['Tag "New" cannot be renamed to "a]b": it contains "]", which ends a [] tag in "[Old] [New] x".']


@pytest.mark.parametrize("name", ["[Old] (Old) x", "(Old) [Old] x"])
def test_a_mixed_chain_refusal_names_the_bracket_that_fails(tmp_path: Path, monkeypatch, name: str) -> None:
    """The refusal names the segment the new tag breaks, not the first one
    carrying the old tag: "a)b" is fine inside [] and only fails inside ()."""
    from descape import viewer as viewer_module

    warnings = []
    monkeypatch.setattr(viewer_module.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _tagged_window(tmp_path, [name, "plain", "plain", "plain"])
    try:
        window.rename_trigger_tag("Old", "a)b")
        assert window.edit_history.records == []
        assert _names(window)[0] == name
    finally:
        window.close()
    assert warnings == [f'Tag "Old" cannot be renamed to "a)b": it contains ")", which ends a () tag in "{name}".']


def test_a_chain_that_fails_with_no_single_bad_segment_gets_the_generic_reason(tmp_path: Path, monkeypatch) -> None:
    from descape import viewer as viewer_module

    warnings = []
    monkeypatch.setattr(viewer_module.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    monkeypatch.setattr(viewer_module.trigger_organize, "retag_in_chain", lambda name, old, new: "untagged")
    window = _tagged_window(tmp_path, ["[Old] x", "plain", "plain", "plain"])
    try:
        window.rename_trigger_tag("Old", "New")
        assert window.edit_history.records == []
    finally:
        window.close()
    assert warnings == ['Tag "Old" cannot be renamed to "New": it would not read back as that tag in "[Old] x".']


def test_remove_selected_offers_every_tag_in_the_chain(tmp_path: Path, monkeypatch) -> None:
    window = _tagged_window(tmp_path, _CHAINED_NAMES)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1)
        asked = _answer_item(monkeypatch, text="New")
        panel.tag_remove_selected_button.click()
        assert asked == [("Remove tag", "Remove a tag from 1 selected trigger:", ["New", "Old"], False)]
        assert _names(window) == ["[New] setup", "[Old] x", "[Old] solo", "plain"]
        assert window.edit_history.records[-1].touched == [1]
        window.undo()
        assert _names(window) == _CHAINED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_tag_keys_on_the_leading_tag_and_matches_it_anywhere(tmp_path: Path) -> None:
    window = _tagged_window(tmp_path, _CHAINED_NAMES)
    try:
        panel = window.trigger_panel
        assert not _action(panel.trigger_context_menu(panel._item_for_index[3]), "Select Tag").isEnabled()
        _action(panel.trigger_context_menu(panel._item_for_index[0]), "Select Tag").trigger()
        assert panel.selected_trigger_indices() == [0, 1]
        _action(panel.trigger_context_menu(panel._item_for_index[1]), "Select Tag").trigger()
        assert panel.selected_trigger_indices() == [1, 2]
    finally:
        window.close()


def test_a_tag_chained_on_by_a_rename_joins_the_dropdown_at_once(tmp_path: Path) -> None:
    """The leading tag is unchanged, so only a chain comparison notices."""
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        _rename(panel, 1, "[d1] [Fresh] armour split")
        assert panel.tag_combo.findData("Fresh") >= 0
        panel.set_tag_filter("Fresh")
        assert _visible_ids(panel) == [1]
        window.undo()
        assert panel.tag_combo.findData("Fresh") < 0
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- navigation affordances: the Sections menu ---------------------------------


def _sectioned_window():
    """The trigger fixture grown to 8 triggers and split into four sections:
    (before) 0, 1 | --RISK!!!-- 2: 3 | --RISK!!!-- 4: 5 | ------ 6: 7. The
    repeated title is the census's duplicate-divider shape."""
    window = _triggers_window()
    for _ in range(4):
        window.trigger_structural_edit("new", [])
    panel = window.trigger_panel
    _rename(panel, 5, "needle target")
    _rename(panel, 2, "--RISK!!!--")
    _rename(panel, 4, "--RISK!!!--")
    _rename(panel, 6, "------")
    assert [s.header_index for s in panel._sections] == [None, 2, 4, 6], "fixture assumption"
    panel.select_trigger(0)
    return window


def _section_actions(panel) -> list[tuple[str, int]]:
    return [(a.text(), a.data()) for a in panel._section_actions]


def _expanded(panel) -> list[bool]:
    return [panel.tree.topLevelItem(i).isExpanded() for i in range(panel.tree.topLevelItemCount())]


def test_the_sections_button_lists_every_section_and_hides_when_flat() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert panel.sections_button.isHidden(), "a flat tree has nothing to collapse or jump to"
        window.close()
        window = _sectioned_window()
        panel = window.trigger_panel
        assert not panel.sections_button.isHidden()
        texts = [a.text() for a in panel.sections_menu.actions() if not a.isSeparator()]
        assert texts[:2] == ["Collapse All", "Expand All"]
        assert _section_actions(panel) == [
            ("(before the first section)", 0),
            ("--RISK!!!--", 2),
            ("--RISK!!!--", 4),
            ("(unnamed section)", 6),
        ]
        panel.sort_combo.setCurrentIndex(1)  # File order
        assert panel.sections_button.isHidden()
    finally:
        window.edit_history.mark_saved()
        window.close()


def _roomy_tree(panel) -> None:
    """Give the trigger tree pane 500 px of the vertical split. At the default
    1500x900 split five button rows leave the tree ~3 rows, so a test that
    clicks a real row position needs the room to see it."""
    from PyQt5.QtWidgets import QApplication

    sizes = panel.splitter.sizes()
    panel.splitter.setSizes([500, max(sum(sizes) - 500, 1)])
    QApplication.processEvents()


def test_clicking_the_synthetic_header_shows_no_trigger() -> None:
    """GH #1: the "(before the first section)" row is not a trigger, so a click
    on it leaves the entry tree empty rather than showing a stale trigger."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _roomy_tree(panel)
        _rename(panel, 2, "--- Second ---")  # sections: (before) 0, 1 | 2: 3
        panel.select_trigger(1)
        assert panel.entry_tree.topLevelItemCount() > 0, "fixture assumption: a trigger is showing"

        synthetic = panel.tree.topLevelItem(0)
        assert synthetic.data(0, Qt.UserRole) is None, "fixture assumption: row 0 is the synthetic header"
        pos = panel.tree.visualItemRect(synthetic).center()
        QTest.mouseClick(panel.tree.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
        QApplication.processEvents()
        assert panel.tree.currentItem() is synthetic, "the click did not land on the header"
        assert panel.current_trigger_index() is None
        assert panel.entry_tree.topLevelItemCount() == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_the_only_divider_back_folds_the_tree_flat() -> None:
    """GH #1: undoing the divider by hand, not by Undo, drops the grouping."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 2, "--- Second ---")
        assert not panel.sections_button.isHidden(), "fixture assumption: the rename grouped the tree"

        _rename(panel, 2, "Fixture: references")
        assert panel.sections_button.isHidden()
        assert all(panel.tree.topLevelItem(i).childCount() == 0 for i in range(panel.tree.topLevelItemCount()))
        assert _trigger_row_count(panel.tree) == panel.tree.topLevelItemCount() == 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_collapsed_section_survives_a_rename_rebuild() -> None:
    """Collapse the second of two same-titled sections, then rename a
    different trigger across the divider predicate, which rebuilds the tree
    through _changed()'s escalation. Only that one section stays collapsed."""
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel._item_for_index[4].setExpanded(False)
        assert panel._collapsed_keys == {("--RISK!!!--", 1)}

        _rename(panel, 7, "--- Last ---")
        assert [s.header_index for s in panel._sections] == [None, 2, 4, 6, 7], "the rename rebuilt the tree"
        assert not panel._item_for_index[4].isExpanded()
        assert panel._item_for_index[2].isExpanded(), "its same-titled twin is a different section"
        assert panel._item_for_index[6].isExpanded()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_collapse_all_survives_a_rebuild_and_expand_all_undoes_it() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)  # a header: selecting it expands no parent
        panel.collapse_all_action.trigger()
        assert _expanded(panel) == [False] * 4
        panel.show_scenario(panel._loaded, panel._pending_exec_order)
        assert _expanded(panel) == [False] * 4

        panel.expand_all_action.trigger()
        assert panel._collapsed_keys == set()
        panel.show_scenario(panel._loaded, panel._pending_exec_order)
        assert _expanded(panel) == [True] * 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_jump_selects_the_divider_or_the_first_member_and_expands_it() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.collapse_all_action.trigger()
        panel._section_actions[2].trigger()
        assert panel.current_trigger_index() == 4
        assert panel.selected_trigger_indices() == [4]
        assert panel._item_for_index[4].isExpanded()
        assert ("--RISK!!!--", 1) not in panel._collapsed_keys, "a jumped-to section stays open on rebuild"

        panel._section_actions[0].trigger()
        assert panel.current_trigger_index() == 0, "the synthetic section's first member"
        assert panel.tree.topLevelItem(0).isExpanded()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_filter_reveals_a_match_in_a_collapsed_section_and_clearing_restores() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel._item_for_index[4].setExpanded(False)
        panel.filter_edit.setText("needle")
        match = panel._item_for_index[5]
        assert not match.isHidden()
        assert match.parent().isExpanded()
        assert panel._collapsed_keys == {("--RISK!!!--", 1)}, "the filter pass records nothing"

        panel.filter_edit.clear()
        assert _expanded(panel) == [True, True, False, True]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_collapse_all_under_an_active_filter_keeps_the_matches_visible() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.filter_edit.setText("needle")
        panel.collapse_all_action.trigger()
        match = panel._item_for_index[5]
        assert not match.isHidden() and match.parent().isExpanded()

        panel.filter_edit.clear()
        assert _expanded(panel) == [False] * 4, "the collapse is what clearing the filter restores"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_jump_to_a_filtered_out_section_is_disabled() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.filter_edit.setText("needle")
        panel.sections_menu.aboutToShow.emit()
        assert [a.isEnabled() for a in panel._section_actions] == [False, False, True, False]
        panel.filter_edit.clear()
        panel.sections_menu.aboutToShow.emit()
        assert all(a.isEnabled() for a in panel._section_actions)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_a_divider_to_another_divider_retitles_its_section() -> None:
    """No repartition, so no rebuild: the menu label and the collapse keys
    have to follow the new title on their own."""
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel._item_for_index[4].setExpanded(False)
        _rename(panel, 2, "--- Renamed ---")
        assert [text for text, _ in _section_actions(panel)][1:3] == ["--- Renamed ---", "--RISK!!!--"]
        assert panel._collapsed_keys == {("--RISK!!!--", 0)}
        panel.select_trigger(0)
        panel.show_scenario(panel._loaded, panel._pending_exec_order)
        assert _expanded(panel) == [True, True, False, True]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_collapse_state_is_dropped_with_the_document() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.collapse_all_action.trigger()
        assert panel._collapsed_keys
        panel.clear_document()
        assert panel._collapsed_keys == set()
        assert panel.sections_button.isHidden()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- section management: New Section, Rename Section, Move to Section -----------


def _order(window) -> list[int]:
    return list(window.trigger_panel._manager().trigger_display_order)


def _record_count(window) -> int:
    return len(window.edit_history.records)


def _divided_window(name: str = "--A--", at: int = 2):
    """The trigger fixture with trigger `at` renamed to divider `name`: by
    default (before) 0, 1 | --A-- 2: 3."""
    window = _triggers_window()
    _rename(window.trigger_panel, at, name)
    window.trigger_panel.select_trigger(0)
    return window


def test_new_section_groups_a_divider_free_file_in_the_default_format() -> None:
    from descape.trigger_organize import is_divider, sections

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert not panel._grouped, "fixture assumption: no dividers"
        before = _record_count(window)
        window.trigger_structural_edit("new_section", [1], "Later")
        names = _names(window)
        assert len(names) == 5 and names[4] == "--- Later ---"
        assert is_divider(names[4])
        assert _order(window) == [0, 1, 2, 3, 4], "the only section's end is the end"
        assert [s.header_index for s in sections(names, _order(window))] == [None, 4]
        assert panel._grouped
        assert panel.current_trigger_index() == 4
        assert _record_count(window) == before + 1
        assert window.trigger_edits.has_edits
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_new_section_lands_after_the_current_section_in_the_files_own_format() -> None:
    window = _divided_window()
    try:
        window.trigger_structural_edit("new_section", [0], "Middle")
        assert _names(window)[4] == "--Middle--", "the file's only titled divider is tight 2+2"
        assert _order(window) == [0, 1, 4, 2, 3]
        window.trigger_structural_edit("new_section", [], "Last")
        assert _order(window)[-1] == 5
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_new_section_leaves_collapsed_sections_collapsed() -> None:
    """GH #135: the rebuild restored the old current trigger with autoScroll
    on, expanding its collapsed section just before the new divider took over."""
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.select_triggers([5, 3])
        panel.collapse_all_action.trigger()
        keys = {("", 0), ("--RISK!!!--", 0), ("--RISK!!!--", 1), ("------", 0)}
        assert panel._collapsed_keys == keys
        assert _expanded(panel) == [False] * 4

        window.trigger_structural_edit("new_section", [5], "Z")
        assert [s.header_index for s in panel._sections] == [None, 2, 4, 8, 6]
        assert panel.current_trigger_index() == 8
        assert panel._collapsed_keys == keys
        assert _expanded(panel) == [False, False, False, True, False], "only the new section is open"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_one_undo_reverses_the_whole_new_section_gesture() -> None:
    window = _divided_window()
    try:
        order = _order(window)
        window.trigger_structural_edit("new_section", [0], "Middle")
        window.undo()
        assert len(_names(window)) == 4
        assert _order(window) == order
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_empty_or_unreadable_title_records_nothing(monkeypatch) -> None:
    from descape import viewer

    warned = []
    monkeypatch.setattr(viewer.QMessageBox, "warning", staticmethod(lambda *args: warned.append(args)))
    window = _divided_window()
    try:
        before = _record_count(window)
        window.trigger_structural_edit("new_section", [0], "   ")
        assert not warned
        window.trigger_structural_edit("new_section", [0], "ends in a dash-")
        assert len(warned) == 1
        assert _record_count(window) == before and len(_names(window)) == 4
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_to_section_lands_at_the_end_of_that_section() -> None:
    window = _divided_window()
    try:
        panel = window.trigger_panel
        before = _record_count(window)
        window.trigger_structural_edit("move_to_section", [0], 2)
        assert _order(window) == [1, 2, 3, 0]
        assert panel.selected_trigger_indices() == [0]
        assert _record_count(window) == before + 1
        assert window.trigger_edits.has_edits, "a pure permutation must still dirty the model"
        window.trigger_structural_edit("move_to_section", [3], None)
        assert _order(window) == [1, 3, 2, 0]
        window.undo()
        assert _order(window) == [1, 2, 3, 0]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_move_to_section_moves_a_selection_as_one_block() -> None:
    window = _divided_window(at=3)
    try:
        window.trigger_structural_edit("new_section", [3], "B")
        assert _order(window) == [0, 1, 2, 3, 4]
        window.trigger_structural_edit("move_to_section", [2, 0], 4)
        assert _order(window) == [1, 3, 4, 0, 2]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_move_that_changes_nothing_or_moves_a_header_records_nothing() -> None:
    window = _divided_window()
    try:
        before = _record_count(window)
        order = _order(window)
        window.trigger_structural_edit("move_to_section", [3], 2)
        window.trigger_structural_edit("move_to_section", [2], None)
        window.trigger_structural_edit("move_to_section", [0], 3)
        assert _record_count(window) == before
        assert _order(window) == order
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_empty_leading_section_is_a_move_target_at_slot_zero() -> None:
    window = _divided_window(at=0)
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        panel._populate_move_menu()
        actions = panel.section_move_menu.actions()
        assert [(a.text(), a.data(), a.isEnabled()) for a in actions] == [
            ("(before the first section)", None, True),
            ("--A--", 0, False),
        ]
        actions[0].trigger()
        assert _order(window) == [2, 0, 1, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_move_menu_keys_on_the_header_so_duplicate_titles_stay_apart() -> None:
    window = _sectioned_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        panel._populate_move_menu()
        assert [(a.text(), a.data(), a.isEnabled()) for a in panel.section_move_menu.actions()] == [
            ("(before the first section)", None, False),
            ("--RISK!!!--", 2, True),
            ("--RISK!!!--", 4, True),
            ("(unnamed section)", 6, True),
        ]
        panel.section_move_menu.actions()[2].trigger()
        assert _order(window) == [1, 2, 3, 4, 5, 0, 6, 7]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rename_section_keeps_the_decoration_and_retitles_the_section() -> None:
    window = _divided_window("--- Old ----")
    try:
        panel = window.trigger_panel
        before = _record_count(window)
        panel.select_trigger(2)
        panel.rename_section(2, "New")
        assert _names(window)[2] == "--- New ----"
        assert [s.title for s in panel._sections if s.header_index == 2] == ["--- New ----"]
        assert _record_count(window) == before + 1
        panel.rename_section(2, "New")
        assert _record_count(window) == before + 1, "an unchanged title records nothing"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rename_section_refuses_a_title_that_would_not_read_back(monkeypatch) -> None:
    from descape import trigger_panel

    warned = []
    monkeypatch.setattr(trigger_panel.QMessageBox, "warning", staticmethod(lambda *args: warned.append(args)))
    window = _divided_window()
    try:
        before = _record_count(window)
        window.trigger_panel.rename_section(2, "x-")
        assert warned and _names(window)[2] == "--A--"
        assert _record_count(window) == before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_section_dialogs_report_through_the_funnels(monkeypatch) -> None:
    window = _divided_window()
    try:
        panel = window.trigger_panel
        asked = _answer_dialogs(monkeypatch, text="  Via dialog ")
        panel.request_new_section()
        assert _names(window)[4] == "--Via dialog--"
        panel.select_trigger(2)
        panel.request_section_rename()
        assert asked["getText"][-1][2] == "A", "prefilled with the current title"
        assert _names(window)[2] == "--Via dialog--"
        _answer_dialogs(monkeypatch, text=None)
        count = _record_count(window)
        panel.request_new_section()
        panel.request_section_rename()
        assert _record_count(window) == count, "a cancelled dialog records nothing"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_section_verbs_are_gated_like_move_up_and_down() -> None:
    window = _divided_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        assert panel.section_new_button.isEnabled()
        assert panel.section_move_button.isEnabled()
        assert not panel.section_rename_action.isEnabled(), "not a divider"
        panel.select_trigger(2)
        assert panel.section_rename_action.isEnabled()
        assert not panel.section_move_button.isEnabled(), "a header does not move into a section"
        assert "section header" in panel.section_move_button.toolTip()

        panel.select_trigger(0)
        panel.filter_edit.setText("a")
        for enabled in (panel.section_new_button.isEnabled(), panel.section_move_button.isEnabled()):
            assert not enabled
        assert "clear the filter" in panel.section_new_button.toolTip()
        panel.filter_edit.clear()

        panel.sort_combo.setCurrentIndex(1)
        panel.select_trigger(2)
        assert not panel.section_new_button.isEnabled()
        assert not panel.section_rename_action.isEnabled()
        assert "Display order" in panel.section_move_button.toolTip()
        panel.sort_combo.setCurrentIndex(0)
        assert panel.section_new_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_section_verbs_follow_the_tag_facet_and_a_single_section_file(tmp_path: Path) -> None:
    window = _tagged_window(tmp_path)
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        assert panel.section_new_button.isEnabled()
        assert not panel.section_move_button.isEnabled(), "one section: nowhere else to go"
        assert "no other section" in panel.section_move_button.toolTip()
        panel.set_tag_filter("D1")
        assert not panel.section_new_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rename_is_off_on_a_bare_run_divider() -> None:
    window = _divided_window("------")
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        assert not panel.section_rename_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def _saved_body(path: Path) -> bytes:
    from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

    from descape.scenario_io import load_map_and_units

    return _decompress_bytes(path.read_bytes()[len(load_map_and_units(path).header_bytes) :])


def _save_divided_fixture(tmp_path: Path) -> Path:
    """The fixture with trigger 2 saved as "--A--", as a real file."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = TriggerEditModel(loaded)
    model.manager().triggers[2].name = "--A--"
    model.mark_dirty(2)
    base = tmp_path / "divided.aoe2scenario"
    write_scenario(loaded, base, triggers=model)
    return base


def _open(path: Path):
    window = _window()
    window.load_scenario(path)
    window.mode_combo.setCurrentText("Triggers")
    return window


def test_a_saved_move_to_section_changes_only_the_display_order_bytes(tmp_path: Path) -> None:
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    base = _save_divided_fixture(tmp_path)
    window = _open(base)
    try:
        window.trigger_structural_edit("move_to_section", [0], 2)
        edited = tmp_path / "moved.aoe2scenario"
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)
        regions = window.trigger_edits.regions
    finally:
        window.edit_history.mark_saved()
        window.close()
    before, after = _saved_body(base), _saved_body(edited)
    assert len(before) == len(after)
    diffs = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert diffs, "the move changed no byte"
    units_end = load_map_and_units(base).units_section_end
    start, end = units_end + regions.triggers_end, units_end + regions.display_order_end
    assert all(start <= i < end for i in diffs), "a byte outside the display-order array changed"
    reloaded = _open(edited)
    try:
        assert _order(reloaded) == [1, 2, 3, 0]
    finally:
        reloaded.close()


def test_a_saved_new_section_round_trips(tmp_path: Path) -> None:
    from descape.scenario_write import write_scenario

    base = _save_divided_fixture(tmp_path)
    window = _open(base)
    try:
        window.trigger_structural_edit("new_section", [0], "Middle")
        edited = tmp_path / "new_section.aoe2scenario"
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)
    finally:
        window.edit_history.mark_saved()
        window.close()
    reloaded = _open(edited)
    try:
        names = _names(reloaded)
        assert len(names) == 5 and names[4] == "--Middle--"
        assert _order(reloaded) == [0, 1, 4, 2, 3]
        assert [s.header_index for s in reloaded.trigger_panel._sections] == [None, 4, 2]
    finally:
        reloaded.close()


def test_a_saved_retitle_leaves_every_other_trigger_blob_alone(tmp_path: Path) -> None:
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    base = _save_divided_fixture(tmp_path)
    window = _open(base)
    try:
        window.trigger_panel.rename_section(2, "Longer title")
        edited = tmp_path / "retitled.aoe2scenario"
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)
    finally:
        window.edit_history.mark_saved()
        window.close()
    before = TriggerEditModel(load_map_and_units(base))._blobs
    after_model = TriggerEditModel(load_map_and_units(edited))
    assert after_model.manager().triggers[2].name == "--Longer title--"
    after = after_model._blobs
    assert [b for i, b in enumerate(before) if i != 2] == [b for i, b in enumerate(after) if i != 2]
    assert before[2] != after[2]


@pytest.mark.corpus
def test_a_new_section_on_a_real_scenario_matches_its_format_and_survives_save(
    scenario_path, tmp_path
) -> None:
    from descape.scenario_write import write_scenario
    from descape.trigger_organize import divider_format, format_divider, is_divider, sections

    window = _open(scenario_path)
    try:
        if not window.scenario.trigger_read_supported or not window.scenario.trigger_write_supported:
            pytest.skip(f"{scenario_path.name}: triggers are not writable")
        panel = window.trigger_panel
        if panel._manager() is None or not panel._manager().triggers:
            pytest.skip(f"{scenario_path.name}: no triggers")
        expected = format_divider("DEscape probe", divider_format(_names(window)))
        count = len(_names(window))
        window.trigger_structural_edit("new_section", [_order(window)[0]], "DEscape probe")
        assert _names(window)[count] == expected
        edited = tmp_path / scenario_path.name
        write_scenario(window.scenario, edited, triggers=window.trigger_edits)
    finally:
        window.edit_history.mark_saved()
        window.close()
    reloaded = _open(edited)
    try:
        names, order = _names(reloaded), _order(reloaded)
        assert len(names) == count + 1 and names[count] == expected and is_divider(expected)
        assert count in [s.header_index for s in sections(names, order)]
    finally:
        reloaded.close()


# -- multi-select entries (GH #60) ---------------------------------------------

# The fixture's "armour split" trigger: one condition, two Modify Attribute effects.
_ARMOUR_TRIGGER = 1


def _select_entries(panel, *refs) -> None:
    """Select these (kind, index) entry rows, Qt's Ctrl+click minus the mouse."""
    from PyQt5.QtWidgets import QApplication

    panel.select_entries(list(refs))
    QApplication.processEvents()


def _entry_counts(window, trigger_index: int) -> tuple[int, int]:
    trigger = window.trigger_panel._manager().triggers[trigger_index]
    return (len(trigger.conditions), len(trigger.effects))


def _effect_summary(panel, trigger_index: int, entry_index: int) -> tuple:
    """A comparable summary of one effect, so a delete can be checked to have
    removed the rows it was given rather than merely that many rows."""
    effect = panel._manager().triggers[trigger_index].effects[entry_index]
    return (effect.effect_type, panel._describe("effect", effect))


def test_selected_entry_refs_reads_in_tree_order_and_drops_non_entries() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 1), ("condition", 0), ("effect", 0))
        assert panel.selected_entry_refs() == [("condition", 0), ("effect", 0), ("effect", 1)]
        assert panel.current_entry_ref() == ("effect", 1), "the first ref is the current row"

        # The trigger row and both group headings are not entries.
        for top in range(3):
            panel.entry_tree.topLevelItem(top).setSelected(True)
        assert panel.selected_entry_refs() == [("condition", 0), ("effect", 0), ("effect", 1)]

        # A hidden row never reaches a bulk verb.
        panel._entry_item_for("effect", 0).setHidden(True)
        assert panel.selected_entry_refs() == [("condition", 0), ("effect", 1)]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_form_of_two_entries_is_the_fields_they_share() -> None:
    """The armour-split pair decides the rule: both are Modify Attribute, so
    every name matches, but their live cluster slots differ and the cluster
    fields carry different read_only flags. Those must drop out."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 0))
        alone = [s.name for s, *_ in panel._rows]
        cluster = {"quantity", "armour_attack_quantity", "armour_attack_class"}
        assert cluster <= set(alone)

        _select_entries(panel, ("effect", 0), ("effect", 1))
        shared = [s.name for s, *_ in panel._rows]
        assert not cluster & set(shared)
        assert shared == [name for name in alone if name in shared], "the first entry's order"
        assert {"object_list_unit_id", "source_player", "operation", "object_attributes"} <= set(shared)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_two_entry_types_intersect_to_what_they_genuinely_share() -> None:
    """A mixed selection is not refused: it shows what the types really have
    in common, and says so when that is nothing."""
    from PyQt5.QtWidgets import QLabel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)  # Activate Trigger + Deactivate Trigger
        _select_entries(panel, ("effect", 0), ("effect", 1))
        assert [s.name for s, *_ in panel._rows] == ["trigger_id"]
        combo = _row_widget(panel, "trigger_id")
        assert combo.currentText() == "(differs)", "they point at different triggers"

        panel.select_trigger(0)
        _select_entries(panel, ("condition", 0), ("effect", 0))
        assert not panel._rows
        row = panel.property_form.itemAt(0).widget()
        assert isinstance(row, QLabel)
        assert row.text() == "No fields are shared by the 2 selected entries."
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_multi_entry_walk_records_nothing_and_saves_byte_identically(tmp_path: Path) -> None:
    """test_browsing_every_row_saves_byte_identically, extended to the set:
    pairing every entry row with every other one builds no model."""
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        for i in range(panel.tree.topLevelItemCount()):
            panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
            QApplication.processEvents()
            refs = [
                (kind, c) for kind in ("condition", "effect") for c in range(_group(panel, kind).childCount())
            ]
            for first in refs:
                for second in refs:
                    _select_entries(panel, first, second)

        assert window.trigger_edits is None, "browsing a selection must not build an edit model"
        assert not window.edit_history.is_dirty
        out = tmp_path / "walked.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_one_selection_change_populates_the_form_once() -> None:
    """A click fires both entry-tree signals; without the latch every click
    builds the form twice, which no widget assertion notices."""
    from PyQt5.QtCore import QItemSelectionModel
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        QApplication.processEvents()
        calls = []
        original = panel._populate_property_form
        panel._populate_property_form = lambda: (calls.append(1), original())[1]

        effects = _group(panel, "effect")
        panel.entry_tree.setCurrentItem(
            effects.child(0), 0, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        )
        QApplication.processEvents()
        assert len(calls) == 1

        effects.child(1).setSelected(True)
        QApplication.processEvents()
        assert len(calls) == 2, "adding a row to the set rebuilds the form once"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_entry_selection_set_survives_a_rebuild() -> None:
    """show_scenario() rebuilds both trees (every undo does), so the set is
    restored whole, not just its current row."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 1), ("condition", 0))
        panel.show_scenario(window.scenario)
        assert panel.selected_entry_refs() == [("condition", 0), ("effect", 1)]
        assert panel.current_entry_ref() == ("effect", 1)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_and_delete_follow_the_set_while_type_stays_single() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 0))
        assert panel.entry_copy_button.isEnabled()
        assert panel.entry_retype_button.isEnabled()

        _select_entries(panel, ("effect", 0), ("effect", 1))
        assert panel.entry_copy_button.isEnabled()
        assert panel.entry_delete_button.isEnabled()
        assert not panel.entry_retype_button.isEnabled(), "the picker is one entry's own kind"

        _select_entries(panel)  # nothing resolvable: the selection is unchanged
        panel.entry_tree.clearSelection()
        assert not panel.entry_copy_button.isEnabled()
        assert not panel.entry_delete_button.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_deleting_two_entries_is_one_record_that_undoes_whole() -> None:
    """Descending removal is what keeps the second index pointing at the row
    the user picked once the first is gone."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        index = 0
        panel.select_trigger(index)
        window.entry_structural_edit("copy", index, "effect", 0, -1)
        window.entry_structural_edit("new", index, "effect", -1, 8)  # ACTIVATE_TRIGGER
        window.edit_history.mark_saved()
        before = _record_count(window)
        conditions, effects = _entry_counts(window, index)
        assert effects == 3
        kept = _effect_summary(panel, index, 1)

        _select_entries(panel, ("effect", 0), ("effect", 2))
        panel.entry_delete_button.click()
        assert _entry_counts(window, index) == (conditions, effects - 2)
        assert _record_count(window) == before + 1, "N deletes are one undo step"
        assert window.edit_history.records[-1].label == "Delete 2 entries"
        assert _effect_summary(panel, index, 0) == kept
        assert panel.selected_entry_refs() == [("effect", 0)], "lands on the lowest removed slot, clamped"

        window.undo()
        assert _entry_counts(window, index) == (conditions, effects)
        window.redo()
        assert _entry_counts(window, index) == (conditions, effects - 2)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copying_two_entries_appends_both_and_selects_them() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        before = _record_count(window)
        conditions, effects = _entry_counts(window, _ARMOUR_TRIGGER)
        originals = [_effect_summary(panel, _ARMOUR_TRIGGER, i) for i in (0, 1)]

        _select_entries(panel, ("effect", 0), ("effect", 1))
        panel.entry_copy_button.click()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions, effects + 2)
        assert _record_count(window) == before + 1, "N copies are one undo step"
        assert window.edit_history.records[-1].label == "Copy 2 entries"
        copies = [_effect_summary(panel, _ARMOUR_TRIGGER, effects + i) for i in (0, 1)]
        assert copies == originals
        assert panel.selected_entry_refs() == [("effect", effects), ("effect", effects + 1)]

        window.undo()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions, effects)
        window.redo()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions, effects + 2)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_selection_spanning_both_lists_deletes_as_one_record() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        before = _record_count(window)
        conditions, effects = _entry_counts(window, _ARMOUR_TRIGGER)

        _select_entries(panel, ("condition", 0), ("effect", 1))
        panel.entry_delete_button.click()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions - 1, effects - 1)
        assert _record_count(window) == before + 1

        window.undo()
        assert _entry_counts(window, _ARMOUR_TRIGGER) == (conditions, effects)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_single_entry_delete_keeps_its_old_label_and_landing() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        _select_entries(panel, ("effect", 1))
        panel.entry_delete_button.click()
        assert window.edit_history.records[-1].label == "Delete effect"
        assert panel.current_entry_ref() == ("effect", 0), "the last row's delete lands above it"
    finally:
        window.edit_history.mark_saved()
        window.close()


_CREATE_OBJECT = 11
_CHANGE_OWNERSHIP = 18
_DISPLAY_INSTRUCTIONS = 20
_ARCHER, _SCOUT = 4, 448


def _spec_named(panel, kind: str, entry, name: str):
    return next(s for s in panel._specs_for(kind, entry) if s.name == name)


def _appended_effects(window, type_id: int, per_entry) -> list[int]:
    """Append one `type_id` effect per dict in `per_entry` to trigger 0, set
    its fields through the real funnel, and return their indices. Built at
    runtime: extending the shared fixture breaks 37 tests pinning its layout."""
    panel = window.trigger_panel
    panel.select_trigger(0)
    start = len(panel._manager().triggers[0].effects)
    for offset, values in enumerate(per_entry):
        window.entry_structural_edit("new", 0, "effect", -1, type_id)
        for name, value in values.items():
            entry = panel._manager().triggers[0].effects[start + offset]
            window.set_entry_field(0, "effect", start + offset, _spec_named(panel, "effect", entry, name), value)
    window.edit_history.mark_saved()
    return list(range(start, start + len(per_entry)))


def _create_objects_window():
    """The issue's own example: three Create Object effects agreeing on unit
    and player, differing only in where they put it."""
    window = _triggers_window()
    rows = [
        {"object_list_unit_id": _ARCHER, "source_player": 1, "location_x": 10 + n, "location_y": 20 + n}
        for n in range(3)
    ]
    return window, _appended_effects(window, _CREATE_OBJECT, rows)


def _effects0(window):
    return window.trigger_panel._manager().triggers[0].effects


def _focus_out_every_row(panel) -> None:
    """Enter and leave every editor, the way tabbing through the form does:
    editingFinished fires on a bare focus-out."""
    from PyQt5.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit, QPlainTextEdit, QWidget

    from descape.value_picker import ValueLineEdit

    for _spec, _kind, _index, widget in panel._rows:
        editors = [widget]
        if not isinstance(widget, (ValueLineEdit, QLineEdit, QPlainTextEdit, QAbstractSpinBox)):
            editors = widget.findChildren(QAbstractSpinBox) or [widget]
        for editor in editors:
            target = editor.line_edit if isinstance(editor, ValueLineEdit) else editor
            if isinstance(target, QWidget) and target.isEnabled():
                target.setFocus()
                QApplication.processEvents()
            if hasattr(target, "editingFinished"):
                target.editingFinished.emit()
            QApplication.processEvents()
    panel.entry_tree.setFocus()
    QApplication.processEvents()


def _hint_texts(panel) -> list[str]:
    from PyQt5.QtWidgets import QLabel

    return [w.text() for w in panel.property_host.findChildren(QLabel) if "Create Objects" in w.text() and w.isVisible()]


def test_a_create_object_effect_names_the_create_objects_tool() -> None:
    """GH #59: the stamping tool works but is hard to find, so the form says so."""
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, ("effect", indices[0]))
        assert _hint_texts(panel) == ["Tip: stamp it with Create Objects."]
        _select_entries(panel, *[("effect", i) for i in indices])
        assert len(_hint_texts(panel)) == 1, "a set of Create Objects shares it"

        _select_entries(panel, ("effect", 0))
        assert _effects0(window)[0].effect_type != _CREATE_OBJECT, "fixture assumption"
        assert _hint_texts(panel) == []
        _select_entries(panel, ("effect", 0), ("effect", indices[0]))
        assert _hint_texts(panel) == [], "only when every selected effect is one"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_differing_field_reads_as_differs_and_a_shared_one_shows_its_value() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QComboBox

    from descape.constant_picker import CatalogLineEdit

    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        unit = _row_widget(panel, "object_list_unit_id")
        assert isinstance(unit, CatalogLineEdit) and unit.value() == _ARCHER
        assert unit.line_edit.placeholderText() == "(unset)"
        player = _row_widget(panel, "source_player")
        assert isinstance(player, QComboBox) and player.currentData() == 1
        assert player.findText("(differs)") < 0, "a shared combo gets no (differs) row"
        location = _row_widget(panel, "location_x")
        assert location.is_indeterminate() and location.lineEdit().text() == ""
        item = _row_widget(panel, "item_id")
        shared = _effects0(window)[indices[0]].item_id
        assert not _reads_differs(item) and item.value() == shared, "a shared value is shown as itself"
        sound = _row_widget(panel, "disable_sound")
        assert sound.checkState() != Qt.PartiallyChecked and not sound.isTristate()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_focusing_every_indeterminate_widget_writes_nothing(tmp_path: Path) -> None:
    """The load-bearing one. Two entries differing in STR, prose, INT, ENUM,
    BOOL, INT_LIST and catalog fields: tab through every row, then save. No
    record, no model, byte-identical output."""
    from PyQt5.QtCore import Qt

    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        indices = _appended_effects(
            window,
            _DISPLAY_INSTRUCTIONS,
            [
                {"object_list_unit_id": _ARCHER, "source_player": 1, "display_time": 5, "play_sound": 0,
                 "message": "hello", "sound_name": "first"},
                {"object_list_unit_id": _SCOUT, "source_player": 2, "display_time": 9, "play_sound": 1,
                 "message": "goodbye", "sound_name": "second"},
            ],
        )
        indices += _appended_effects(
            window,
            _CHANGE_OWNERSHIP,
            [{"flash_object": 0, "selected_object_ids": [1, 2]}, {"flash_object": 1, "selected_object_ids": [3]}],
        )
        baseline = tmp_path / "baseline.aoe2scenario"
        write_scenario(window.scenario, baseline, triggers=window.trigger_edits)
        records = _record_count(window)

        for pair in (indices[:2], indices[2:]):
            _select_entries(panel, *[("effect", i) for i in pair])
            differs = [s.name for s, _k, _i, w in panel._rows if _reads_differs(w)]
            assert len(differs) >= 2, differs
            _focus_out_every_row(panel)

        assert _record_count(window) == records, "a focus-out on a differing field must not write"
        assert not window.edit_history.is_dirty
        _select_entries(panel, *[("effect", i) for i in indices[2:]])
        assert _row_widget(panel, "flash_object").checkState() == Qt.PartiallyChecked
        out = tmp_path / "focused.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == baseline.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def _reads_differs(widget) -> bool:
    """Whether a form widget is in its "(differs)" state, per widget kind."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QCheckBox, QComboBox, QLineEdit, QPlainTextEdit

    from descape.value_picker import ValueLineEdit
    from descape.viewer_common import _IndeterminateMixin

    if isinstance(widget, ValueLineEdit):
        return widget.line_edit.placeholderText() == "(differs)"
    if isinstance(widget, (QLineEdit, QPlainTextEdit)):
        return widget.placeholderText() == "(differs)"
    if isinstance(widget, QComboBox):
        return widget.currentText() == "(differs)"
    if isinstance(widget, QCheckBox):
        return widget.checkState() == Qt.PartiallyChecked
    return isinstance(widget, _IndeterminateMixin) and widget.is_indeterminate()


def test_a_group_edit_writes_every_selected_entry_as_one_record() -> None:
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        before = _record_count(window)
        locations = [(_effects0(window)[i].location_x, _effects0(window)[i].location_y) for i in indices]

        unit = _row_widget(panel, "object_list_unit_id")
        unit.line_edit.setText(str(_SCOUT))
        unit.line_edit.editingFinished.emit()

        assert [_effects0(window)[i].object_list_unit_id for i in indices] == [_SCOUT] * 3
        assert [(_effects0(window)[i].location_x, _effects0(window)[i].location_y) for i in indices] == locations
        assert _record_count(window) == before + 1, "a group edit is one undo step"
        assert window.edit_history.records[-1].label == "Set object list unit id on 3 entries"
        assert panel.selected_entry_refs() == [("effect", i) for i in indices], "the set survives the edit"

        window.undo()
        assert [_effects0(window)[i].object_list_unit_id for i in indices] == [_ARCHER] * 3
        window.redo()
        assert [_effects0(window)[i].object_list_unit_id for i in indices] == [_SCOUT] * 3
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_group_edit_reaches_the_entries_the_current_row_agrees_with() -> None:
    """"At least one selected entry differs", not "the current one differs":
    a value the current row already holds must still reach the others."""
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        entry = _effects0(window)[indices[0]]
        window.set_entry_field(0, "effect", indices[0], _spec_named(panel, "effect", entry, "source_player"), 3)
        window.edit_history.mark_saved()
        _select_entries(panel, *[("effect", i) for i in indices])
        before = _record_count(window)

        player = _row_widget(panel, "source_player")
        assert player.currentText() == "(differs)"
        player.setCurrentIndex(player.findData(3))

        assert [_effects0(window)[i].source_player for i in indices] == [3, 3, 3]
        assert _record_count(window) == before + 1
        assert window.edit_history.records[-1].label == "Set source player on 2 entries"
        player.setCurrentIndex(0)  # back to "(differs)": writes nothing
        assert _record_count(window) == before + 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_indeterminate_spinbox_commits_its_first_value_to_every_entry() -> None:
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        spin = _row_widget(panel, "location_x")
        assert spin.is_indeterminate() and spin.lineEdit().text() == ""

        spin.setValue(30)
        assert not spin.is_indeterminate()
        assert spin.lineEdit().text() == "30", "the committed value has to become visible"
        assert [_effects0(window)[i].location_x for i in indices] == [30, 30, 30]
        assert [_effects0(window)[i].location_y for i in indices] == [20, 21, 22]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_typing_the_parked_value_into_a_blank_spinbox_still_commits() -> None:
    """The spinbox parks on -1 (UNSET), so typing -1 emits no valueChanged.
    Group-setting a location back to unset must still work."""
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        spin = _row_widget(panel, "location_x")
        before = _record_count(window)
        # A bare focus-out first (this offscreen window never moves focus, so
        # it is emitted): nothing typed, nothing written.
        spin.editingFinished.emit()
        assert _record_count(window) == before
        QTest.keyClicks(spin.lineEdit(), "-1")
        spin.editingFinished.emit()
        QApplication.processEvents()
        assert [_effects0(window)[i].location_x for i in indices] == [-1, -1, -1]
        assert spin.lineEdit().text() == "(unset)"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_tristate_checkbox_and_a_blank_list_write_only_once_changed() -> None:
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        indices = _appended_effects(
            window,
            _CHANGE_OWNERSHIP,
            [{"flash_object": 0, "selected_object_ids": [1, 2]}, {"flash_object": 1, "selected_object_ids": [3]}],
        )
        _select_entries(panel, *[("effect", i) for i in indices])
        before = _record_count(window)
        flash = _row_widget(panel, "flash_object")
        ids = _row_widget(panel, "selected_object_ids")
        assert flash.checkState() == Qt.PartiallyChecked
        assert ids.text() == "" and ids.placeholderText() == "(differs)"

        # insert(), not setText(): setText clears isModified, the flag that
        # separates "typed in" from "merely focused".
        ids.insert("7, 8")
        ids.editingFinished.emit()
        assert [list(_effects0(window)[i].selected_object_ids) for i in indices] == [[7, 8], [7, 8]]

        flash.setCheckState(Qt.Checked)
        assert [_effects0(window)[i].flash_object for i in indices] == [1, 1]
        assert not flash.isTristate(), "the third state is not a value the user can pick"
        assert _record_count(window) == before + 2
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_blank_text_field_writes_only_once_it_is_typed_in() -> None:
    """Pins the accepted limitation too: a differing text field cannot be
    group-cleared, since an empty box means "leave these alone"."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        indices = _appended_effects(
            window,
            _DISPLAY_INSTRUCTIONS,
            [{"sound_name": "first", "message": "hello"}, {"sound_name": "second", "message": "goodbye"}],
        )
        _select_entries(panel, *[("effect", i) for i in indices])
        before = _record_count(window)
        sound = _row_widget(panel, "sound_name")
        message = _row_widget(panel, "message")
        assert sound.text() == "" and sound.placeholderText() == "(differs)"
        assert message.toPlainText() == "" and message.placeholderText() == "(differs)"

        sound.insert("third")
        sound.editingFinished.emit()
        assert [_effects0(window)[i].sound_name for i in indices] == ["third", "third"]

        message.setPlainText("shared")
        message.editingFinished.emit()
        assert [_effects0(window)[i].message for i in indices] == ["shared", "shared"]
        assert _record_count(window) == before + 2
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_single_entry_text_field_still_commits_on_a_plain_focus_out() -> None:
    """N = 1 keeps the pre-#60 rule: the isModified() gate is multi-only."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        indices = _appended_effects(window, _DISPLAY_INSTRUCTIONS, [{"sound_name": "first"}])
        _select_entries(panel, ("effect", indices[0]))
        sound = _row_widget(panel, "sound_name")
        sound.setText("typed by the test")  # leaves isModified() False
        sound.editingFinished.emit()
        assert _effects0(window)[indices[0]].sound_name == "typed by the test"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_widget_built_for_another_selection_writes_nothing() -> None:
    """A focus-out can land after the click that moved the selection; the
    widget's captured refs are then stale and must not write anywhere."""
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", i) for i in indices])
        spec = _spec_named(panel, "effect", _effects0(window)[indices[0]], "location_x")
        stale = [("effect", i) for i in indices]
        _select_entries(panel, ("effect", indices[0]))
        before = _record_count(window)
        panel._changed(spec, stale, 99)
        assert _record_count(window) == before
        assert [_effects0(window)[i].location_x for i in indices] == [10, 11, 12]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_same_type_selects_every_entry_of_that_type() -> None:
    """The issue's literal ask: one Create Object effect in, all of them out,
    the clicked one current. A selection expander only: no model, no record."""
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        before = _record_count(window)
        menu = panel.entry_context_menu(panel._entry_item_for("effect", indices[1]))
        action = _action(menu, "Select Same Type")
        assert action.isEnabled()
        action.trigger()

        assert panel.selected_entry_refs() == [("effect", i) for i in indices]
        assert panel.current_entry_ref() == ("effect", indices[1])
        assert _record_count(window) == before
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_same_type_skips_other_types_and_hidden_rows() -> None:
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        panel._entry_item_for("effect", indices[2]).setHidden(True)
        panel.select_same_type_as(panel._entry_item_for("effect", indices[0]))
        # Effect 0 is the fixture's Display Instructions: another type.
        assert panel.selected_entry_refs() == [("effect", indices[0]), ("effect", indices[1])]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_same_type_is_offered_only_on_an_entry_row() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_ARMOUR_TRIGGER)
        for top in range(3):  # the trigger row and the two group headings
            menu = panel.entry_context_menu(panel.entry_tree.topLevelItem(top))
            assert not _action(menu, "Select Same Type").isEnabled()
        assert _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Select Same Type").isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #137: right-click Cut/Copy/Paste on triggers, conditions and effects ----


@pytest.fixture
def _no_menu_exec(monkeypatch):
    """A stray context-menu event would exec a modal QMenu and hang the run."""
    from descape import trigger_panel as panel_module

    shown: list = []
    monkeypatch.setattr(panel_module.QMenu, "exec_", lambda self, *a, **k: shown.append(self))
    return shown


def _right_click(tree, item) -> None:
    """A real right-button press and release on `item`'s row, near its left
    edge: an entry row's rect runs far past the 340 px viewport."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    tree.scrollToItem(item)
    QApplication.processEvents()
    rect = tree.visualItemRect(item)
    point = QPoint(rect.left() + 8, rect.center().y())
    assert tree.viewport().rect().contains(point), (rect, tree.viewport().rect())
    QTest.mousePress(tree.viewport(), Qt.RightButton, Qt.NoModifier, point)
    QTest.mouseRelease(tree.viewport(), Qt.RightButton, Qt.NoModifier, point)
    QApplication.processEvents()


@pytest.mark.parametrize("clicked", [3, 1], ids=["unselected row", "member of the selection"])
def test_gh137_slice0_ab_a_right_click_on_the_trigger_tree(_no_menu_exec, clicked: int) -> None:
    """Slice 0 a/b: an unselected row becomes the whole selection; a member
    keeps the selection and only becomes current."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _roomy_tree(panel)
        _ctrl_select(panel, 0, 1, 2)
        _right_click(panel.tree, panel._item_for_index[clicked])
        expected = [3] if clicked == 3 else [0, 1, 2]
        assert panel.selected_trigger_indices() == expected
        assert panel.current_trigger_index() == clicked
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("clicked", [3, 1], ids=["unselected row", "member of the selection"])
def test_gh137_slice0_c_a_right_click_on_the_entry_tree(_no_menu_exec, clicked: int) -> None:
    window, indices = _create_objects_window()
    try:
        panel = window.trigger_panel
        assert indices == [1, 2, 3]
        _select_entries(panel, ("effect", 0), ("effect", 1), ("effect", 2))
        _right_click(panel.entry_tree, panel._entry_item_for("effect", clicked))
        expected = [("effect", 3)] if clicked == 3 else [("effect", 0), ("effect", 1), ("effect", 2)]
        assert panel.selected_entry_refs() == expected
        assert panel.current_entry_ref() == ("effect", clicked)
    finally:
        window.edit_history.mark_saved()
        window.close()


_EDITABLE_WIDGETS = ["line edit", "text box", "spin box", "editable combo"]


def _editable_widget(window, which: str):
    """One focused-able editor of each class Edit > Cut/Copy/Paste must defer to.
    No production combo is editable today, so that one is a stand-in."""
    from PyQt5.QtWidgets import QComboBox

    panel = window.trigger_panel
    panel.select_trigger(0)
    if which == "line edit":
        return _row_widget(panel, "name")
    if which == "text box":
        return _row_widget(panel, "description")
    if which == "spin box":
        return _row_widget(panel, "description_order")
    combo = QComboBox(panel)
    combo.setEditable(True)
    combo.addItem("alpha")
    combo.show()
    return combo


def _editor_text(widget) -> str:
    from PyQt5.QtWidgets import QComboBox, QPlainTextEdit

    if isinstance(widget, QPlainTextEdit):
        return widget.toPlainText()
    if isinstance(widget, QComboBox):
        return widget.lineEdit().text()
    return widget.text()


def _select_all_text(widget) -> None:
    from PyQt5.QtWidgets import QComboBox

    (widget.lineEdit() if isinstance(widget, QComboBox) else widget).selectAll()


@pytest.mark.parametrize("route", ["keybind", "menu"])
@pytest.mark.parametrize("which", _EDITABLE_WIDGETS)
def test_gh137_slice0_g_cut_copy_paste_defer_to_a_focused_editor(which: str, route: str) -> None:
    """Slice 0 g, a hard requirement (wave 10): with an editable widget
    focused, Ctrl+X/C/V and Edit > Cut/Copy/Paste act on its text and never
    on the triggers."""
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    clipboard = QApplication.clipboard()
    try:
        widget = _editable_widget(window, which)
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        assert window.isActiveWindow()
        widget.setFocus()
        QApplication.processEvents()
        focused = QApplication.focusWidget()
        assert focused is widget or widget.isAncestorOf(focused), focused

        def press(verb: str) -> None:
            if route == "keybind":
                key = {"cut": Qt.Key_X, "copy": Qt.Key_C, "paste": Qt.Key_V}[verb]
                QTest.keyClick(QApplication.focusWidget(), key, Qt.ControlModifier)
            else:
                action = getattr(window, f"{verb}_action")
                assert action.isEnabled(), verb
                action.trigger()
            QApplication.processEvents()

        baseline = _editor_text(widget)
        assert baseline, "the editor starts with some text to copy"
        clipboard.setText("")
        _select_all_text(widget)
        press("copy")
        assert clipboard.text() == baseline
        assert window._trigger_clipboard is None, "Copy took the triggers"

        window.copy_triggers()  # a held block, so a stray trigger Paste would show
        count = len(window.trigger_panel._manager().triggers)
        clipboard.setText("42")
        _select_all_text(widget)
        press("paste")
        assert _editor_text(widget) == "42"
        assert len(window.trigger_panel._manager().triggers) == count, "Paste pasted triggers"

        _select_all_text(widget)
        press("cut")
        assert _editor_text(widget) == ""
        assert clipboard.text() == "42"
        assert len(window.trigger_panel._manager().triggers) == count, "Cut cut triggers"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


def _texts(menu) -> list[str]:
    return [action.text() for action in menu.actions()]


def _status(window) -> str:
    return window.status_log.toPlainText()


def test_gh137_the_trigger_menu_offers_cut_copy_paste_first() -> None:
    import dataclasses

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(1)
        menu = panel.trigger_context_menu(panel._item_for_index[1])
        assert _texts(menu) == ["Cut", "Copy", "Paste", "", "Select Section", "Select Tag"]
        assert menu.toolTipsVisible()
        assert _action(menu, "Cut").isEnabled() and _action(menu, "Copy").isEnabled()
        assert not _action(menu, "Paste").isEnabled(), "nothing to paste yet"

        panel.tree.clearSelection()
        menu = panel.trigger_context_menu(panel._item_for_index[1])
        assert not _action(menu, "Cut").isEnabled() and not _action(menu, "Copy").isEnabled()

        panel.select_trigger(0)
        _action(panel.trigger_context_menu(panel._item_for_index[0]), "Copy").trigger()
        assert [t.name for t in window._trigger_clipboard.triggers] == ["Fixture: setup"]
        assert window.trigger_edits is None or not window.trigger_edits.has_edits, "Copy records nothing"
        paste = _action(panel.trigger_context_menu(panel._item_for_index[1]), "Paste")
        assert paste.isEnabled() and paste.toolTip() == (
            "Paste the copied triggers below this one, or below a collapsed section's end"
        )

        window._trigger_clipboard = dataclasses.replace(window._trigger_clipboard, scenario_version="1.56")
        window._sync_trigger_clipboard_state()
        paste = _action(panel.trigger_context_menu(panel._item_for_index[1]), "Paste")
        assert paste.toolTip() == "Copied from a 1.56 scenario; this one is 1.58."
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_task214_edit_menu_trigger_clipboard_tips_say_a_collapsed_section_counts_whole() -> None:
    """Edit > Cut/Copy/Paste Triggers explain GH #134's collapsed-section rule like the panel's own controls."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.copy_triggers()
        window._update_clipboard_actions()
        assert "(a collapsed section cuts whole)" in window.cut_action.toolTip()
        assert "(a collapsed section copies whole)" in window.copy_action.toolTip()
        assert window.paste_action.toolTip().endswith(", or below a collapsed section's end")
        assert panel.trigger_paste_button.toolTip() == panel._PASTE_TOOLTIP
        assert "below a collapsed section's end" in panel._PASTE_TOOLTIP
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_menu_paste_lands_below_the_clicked_row_not_the_current_one() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.copy_triggers()
        panel.select_trigger(3)
        _action(panel.trigger_context_menu(panel._item_for_index[1]), "Paste").trigger()
        assert _order(window) == [0, 1, 4, 2, 3]
        assert panel.selected_trigger_indices() == [4]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_paste_is_disabled_on_the_synthetic_leading_header() -> None:
    from PyQt5.QtCore import Qt

    window = _divided_window()
    try:
        panel = window.trigger_panel
        window.copy_triggers()
        synthetic = panel.tree.topLevelItem(0)
        assert synthetic.data(0, Qt.UserRole) is None
        assert not _action(panel.trigger_context_menu(synthetic), "Paste").isEnabled()
        assert _action(panel.trigger_context_menu(panel._item_for_index[2]), "Paste").isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_cut_is_one_record_and_its_paste_keeps_the_name_of_a_divider() -> None:
    """The divider case of decision 2: a cut header still heads a section once
    pasted. An undo of the cut leaves the inbound link on the original, so
    the paste after it leaves that link alone and says so."""
    from descape.trigger_organize import is_divider

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 0, "--- Setup ---")
        panel.select_trigger(0)
        records = _record_count(window)
        _action(panel.trigger_context_menu(panel._item_for_index[0]), "Cut").trigger()
        manager = panel._manager()
        assert [t.name for t in manager.triggers] == ["Fixture: armour split", "Fixture: references", "Fixture: variable"]
        assert _record_count(window) == records + 1
        assert window.edit_history.records[-1].label == "Cut trigger 0"
        assert window._trigger_clipboard.from_cut
        assert "1 trigger link into the cut triggers will reconnect on Paste." in _status(window)
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[1])] == [-1, 0]
        assert panel.selected_trigger_indices() == [0], "lands where the cut row was"

        window.undo()
        manager = panel._manager()
        assert len(manager.triggers) == 4
        assert window._trigger_clipboard is not None, "the clipboard keeps the block"
        window.status_log.clear()
        panel.select_trigger(3)
        window.paste_triggers()
        manager = panel._manager()
        assert manager.triggers[4].name == "--- Setup ---" and is_divider(manager.triggers[4].name)
        assert panel._grouped and panel._item_for_index[4].parent() is None, "it heads its own section"
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[2])] == [0, 1]
        assert "left 1 trigger link alone" in _status(window)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_cut_then_paste_relinks_and_one_undo_takes_both_back() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.cut_triggers()
        panel.select_trigger(2)
        records = _record_count(window)
        window.paste_triggers()
        manager = panel._manager()
        assert [t.name for t in manager.triggers][-1] == "Fixture: setup", "no (copy) suffix"
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[1])] == [3, 0]
        assert "Pasted 1 trigger: reconnected 1 trigger link into them." in _status(window)
        assert _record_count(window) == records + 1, "the relink rides in the paste's record"

        window.undo()
        manager = panel._manager()
        assert len(manager.triggers) == 3
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[1])] == [-1, 0]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_cut_undo_then_paste_leaves_the_restored_link_alone() -> None:
    """The undo puts the inbound link back on the original, so the cut
    block's paste must not re-point it at the copy, and says it left it."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.cut_triggers()
        window.undo()
        window.status_log.clear()
        window.paste_triggers()
        manager = panel._manager()
        assert manager.triggers[4].name == "Fixture: setup"
        assert [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[2])] == [0, 1]
        status = _status(window)
        assert "left 1 trigger link alone (changed or removed since the cut)" in status, status
        assert "reconnected" not in status
    finally:
        window.edit_history.mark_saved()
        window.close()


_CLEARED_SINCE_COPY = "cleared 1 trigger link (target changed or removed since the copy)"


@pytest.mark.parametrize("edited", [False, True], ids=["untouched target", "target edited and undone"])
def test_gh137_a_same_document_trigger_paste_says_it_cleared_a_link(edited: bool) -> None:
    """Trap 5: an undo swaps in a deep copy of the target, so the copied
    block's link no longer resolves and lands on -1. That must not be silent."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        window.copy_triggers()
        if edited:
            _rename(panel, 0, "Fixture: renamed")
            window.undo()
        window.status_log.clear()
        window.paste_triggers()
        manager = panel._manager()
        assert manager.triggers[4].name == "Fixture: references (copy)"
        refs = [ce.trigger_id for ce in trigger_clipboard_refs(manager.triggers[4])]
        assert refs == ([-1, 1] if edited else [0, 1])
        status = _status(window)
        if edited:
            assert f"Pasted 1 trigger: {_CLEARED_SINCE_COPY}." in status, status
        else:
            assert "cleared" not in status, status
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("edited", [False, True], ids=["untouched target", "target edited and undone"])
def test_gh137_a_same_document_entry_paste_says_it_cleared_a_link(edited: bool) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(2)
        _select_entries(panel, ("effect", 0))
        window.copy_entries_to_clipboard()
        if edited:
            _rename(panel, 0, "Fixture: renamed")
            window.undo()
        panel.select_trigger(3)
        window.status_log.clear()
        window.paste_entries()
        pasted = panel._manager().triggers[3].effects[-1]
        assert pasted.trigger_id == (-1 if edited else 0)
        status = _status(window)
        if edited:
            assert f"Pasted 1 effect: {_CLEARED_SINCE_COPY}." in status, status
        else:
            assert "cleared" not in status, status
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_edit_cut_and_ctrl_x_cut_the_selected_triggers() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        panel = window.trigger_panel
        panel.select_trigger(1)
        panel.tree.setFocus()
        QApplication.processEvents()
        assert window.cut_action.text() == "Cu&t Triggers" and window.cut_action.isEnabled()
        QTest.keyClick(panel.tree, Qt.Key_X, Qt.ControlModifier)
        QApplication.processEvents()
        assert [t.name for t in panel._manager().triggers] == [
            "Fixture: setup", "Fixture: references", "Fixture: variable",
        ]
        assert window._trigger_clipboard.from_cut

        panel.select_trigger(2)
        window.cut_action.trigger()
        assert len(panel._manager().triggers) == 2
        assert panel.selected_trigger_indices() == [1], "the last row cut lands on the one above"

        window.mode_combo.setCurrentText("Terrain")
        assert window.cut_action.text() == "Cu&t Triggers"
        assert not window.cut_action.isEnabled(), "there is no Cut Region"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_failed_cut_leaves_the_clipboard_alone(monkeypatch) -> None:
    """_trigger_edit reports a raised body instead of re-raising, so the block
    is committed only once the triggers are really gone."""
    from descape import viewer as viewer_module

    warnings = []
    monkeypatch.setattr(viewer_module.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        manager = window._ensure_trigger_edits().manager()

        def refuse(_indices):
            raise RuntimeError("refused")

        monkeypatch.setattr(manager, "remove_triggers", refuse)
        window.cut_triggers()
        assert len(warnings) == 1
        assert window._trigger_clipboard is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy order", "display order"])
def test_gh137_pasting_a_cut_block_on_a_legacy_order_file_says_it_runs_last(legacy: bool) -> None:
    from types import SimpleNamespace

    from descape.trigger_model import exec_order_value

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        assert exec_order_value(window.scenario) == 0
        if legacy:
            # Map Options' own route: a pending flag counts, as the status line reads it.
            window._set_exec_order(SimpleNamespace(label="execution order"), 1)
            assert window.trigger_edits.exec_order == 1
        panel.select_trigger(0)
        window.cut_triggers()
        window.paste_triggers()
        said = "the pasted triggers run last wherever they are listed" in _status(window)
        assert said is legacy
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_focused_editor_retitles_edit_cut_copy_paste() -> None:
    """Focus moves retitle the three, a tree-to-editor move included, which
    _on_app_focus_changed's text-box early return used to swallow."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        name = _row_widget(panel, "name")
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        panel.tree.setFocus()
        QApplication.processEvents()
        assert [a.text() for a in (window.cut_action, window.copy_action, window.paste_action)] == [
            "Cu&t Triggers", "&Copy Triggers", "&Paste Triggers",
        ]
        assert not window.paste_action.isEnabled(), "no triggers to paste"

        name.setFocus()
        QApplication.processEvents()
        assert [a.text() for a in (window.cut_action, window.copy_action, window.paste_action)] == [
            "Cu&t", "&Copy", "&Paste",
        ]
        assert all(a.isEnabled() for a in (window.cut_action, window.copy_action, window.paste_action))
        assert window.cut_action.toolTip() == "Cut the selected text"

        name.setReadOnly(True)
        panel.tree.setFocus()
        QApplication.processEvents()
        name.setFocus()
        QApplication.processEvents()
        assert window.copy_action.isEnabled()
        assert not window.cut_action.isEnabled() and not window.paste_action.isEnabled(), "read-only text"

        panel.tree.setFocus()
        QApplication.processEvents()
        assert window.cut_action.text() == "Cu&t Triggers"
    finally:
        window.edit_history.mark_saved()
        window.close()


# The fixture's "references" (2): conditions [10], effects [8, 9];
# "variable" (3): conditions [10, 5, 6, 6], effects [56, 19, 11, 14, 15, 12, 19].


def _types_of(window, trigger_index: int, kind: str = "effect") -> list[int]:
    trigger = window.trigger_panel._manager().triggers[trigger_index]
    return [getattr(e, f"{kind}_type") for e in getattr(trigger, f"{kind}s")]


def test_gh137_the_entry_menu_offers_cut_copy_paste_with_its_own_clipboard() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 1), ("effect", 2))
        menu = panel.entry_context_menu(panel._entry_item_for("effect", 1))
        assert _texts(menu) == ["Cut", "Copy", "Paste", "", "Select Same Type"]
        assert menu.toolTipsVisible()
        assert _action(menu, "Cut").isEnabled() and _action(menu, "Copy").isEnabled()
        assert not _action(menu, "Paste").isEnabled(), "nothing to paste yet"

        _action(menu, "Copy").trigger()
        assert window._entry_clipboard.label == "2 effects"
        assert window.trigger_edits is None or not window.trigger_edits.has_edits, "Copy records nothing"
        assert window._trigger_clipboard is None, "two slots: the trigger one is untouched"
        assert not panel.trigger_paste_button.isEnabled(), "the trigger tree's Paste ignores an entry block"
        assert not _action(panel.trigger_context_menu(panel._item_for_index[3]), "Paste").isEnabled()

        panel.select_trigger(2)
        menu = panel.entry_context_menu(panel.entry_tree.topLevelItem(0))
        assert not _action(menu, "Cut").isEnabled(), "no condition or effect selected"
        assert _action(menu, "Paste").isEnabled()
        panel.select_trigger(0)
        window.copy_triggers()
        assert window._entry_clipboard.label == "2 effects", "and copying triggers keeps the entry block"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_entry_paste_lands_below_the_clicked_effect_of_another_trigger() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 1), ("effect", 2))
        window.copy_entries_to_clipboard()
        panel.select_trigger(2)
        records = _record_count(window)
        _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Paste").trigger()
        assert _types_of(window, 2) == [8, 19, 11, 9]
        assert _record_count(window) == records + 1
        assert window.edit_history.records[-1].label == "Paste 2 effects"
        assert panel.selected_entry_refs() == [("effect", 1), ("effect", 2)]

        window.undo()
        assert _types_of(window, 2) == [8, 9]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("row", [0, 1, 2], ids=["trigger row", "conditions heading", "effects heading"])
def test_gh137_entry_paste_on_a_non_entry_row_appends_to_each_list(row: int) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("condition", 1), ("effect", 0))
        window.copy_entries_to_clipboard()
        panel.select_trigger(2)
        _action(panel.entry_context_menu(panel.entry_tree.topLevelItem(row)), "Paste").trigger()
        assert _types_of(window, 2, "condition") == [10, 5]
        assert _types_of(window, 2) == [8, 9, 56]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_cut_entries_is_one_record_and_pastes_into_another_trigger() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("condition", 1), ("effect", 0))
        records = _record_count(window)
        _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Cut").trigger()
        assert (len(_types_of(window, 3, "condition")), len(_types_of(window, 3))) == (3, 6)
        assert _record_count(window) == records + 1
        assert window.edit_history.records[-1].label == "Cut 2 entries"
        assert window._entry_clipboard.label == "1 condition and 1 effect"

        window.undo()
        assert (len(_types_of(window, 3, "condition")), len(_types_of(window, 3))) == (4, 7)
        assert window._entry_clipboard is not None, "the clipboard keeps the block"
        panel.select_trigger(0)
        window.paste_entries()
        assert _types_of(window, 0, "condition") == [10, 5]
        assert _types_of(window, 0) == [20, 56]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_refused_entry_paste_says_why_and_records_nothing() -> None:
    import dataclasses

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 0))
        window.copy_entries_to_clipboard()
        window._entry_clipboard = dataclasses.replace(window._entry_clipboard, scenario_version="1.56")
        window._sync_entry_clipboard_state()
        expected = "Copied from a 1.56 scenario; this one is 1.58."
        paste = _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Paste")
        assert paste.isEnabled() and paste.toolTip() == expected
        records = _record_count(window)
        paste.trigger()
        assert f"Cannot paste these conditions/effects: {expected}" in _status(window)
        assert _record_count(window) == records
        assert len(_types_of(window, 3)) == 7
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_an_entry_paste_into_another_file_reports_and_undoes_in_one_step(tmp_path: Path) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, *[("condition", i) for i in range(4)], *[("effect", i) for i in range(7)])
        window.copy_entries_to_clipboard()
        window.edit_history.mark_saved()
        window.new_map()
        window.mode_combo.setCurrentText("Triggers")
        window.trigger_structural_edit("new", [])
        panel.select_trigger(0)
        window.paste_entries()
        assert len(_types_of(window, 0)) == 7
        assert (
            "Pasted 4 conditions and 7 effects from another scenario: cleared 5 unit references, named 1 variable."
            in _status(window)
        ), _status(window)
        manager = panel._manager()
        assert [(v.variable_id, v.name) for v in manager.variables] == [(0, "fixture_var")]
        out = tmp_path / "pasted.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded_scenario = load_map_and_units(out)
        reloaded = parse_triggers(reloaded_scenario)
        assert len(reloaded.triggers[0].effects) == 7
        assert [(v.variable_id, v.name) for v in reloaded.variables] == [(0, "fixture_var")], "the name is saved"

        window.undo()
        manager = panel._manager()
        assert (len(manager.triggers[0].effects), len(manager.variables)) == (0, 0)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_failed_entry_cut_leaves_the_clipboard_alone(monkeypatch) -> None:
    from descape import viewer as viewer_module

    warnings = []
    monkeypatch.setattr(viewer_module.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 0))
        trigger = window._ensure_trigger_edits().manager().triggers[3]

        def refuse(**_kwargs):
            raise RuntimeError("refused")

        monkeypatch.setattr(trigger, "remove_effect", refuse)
        window.cut_entries()
        assert len(warnings) == 1
        assert window._entry_clipboard is None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_a_repopulate_pushes_the_entry_paste_refusal() -> None:
    import dataclasses

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _select_entries(panel, ("effect", 0))
        window.copy_entries_to_clipboard()
        window._entry_clipboard = dataclasses.replace(window._entry_clipboard, scenario_version="1.56")
        window.mode_combo.setCurrentText("Terrain")
        window.mode_combo.setCurrentText("Triggers")
        panel.select_trigger(3)
        paste = _action(panel.entry_context_menu(panel._entry_item_for("effect", 0)), "Paste")
        assert paste.toolTip() == "Copied from a 1.56 scenario; this one is 1.58."
    finally:
        window.edit_history.mark_saved()
        window.close()


def _active_triggers_window():
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    window.activateWindow()
    QApplication.setActiveWindow(window)
    QApplication.processEvents()
    assert window.isActiveWindow()
    return window


def _focus_on(widget) -> None:
    from PyQt5.QtWidgets import QApplication

    widget.setFocus()
    QApplication.processEvents()
    assert widget.hasFocus()


def _edit_texts(window) -> list[str]:
    return [a.text() for a in (window.cut_action, window.copy_action, window.paste_action)]


def test_gh137_edit_cut_copy_paste_follow_the_focused_tree() -> None:
    """A tree-to-tree focus move retitles the three, and an entry selection
    change re-gates them while the entry tree keeps focus."""
    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _focus_on(panel.entry_tree)
        assert _edit_texts(window) == ["Cu&t Conditions/Effects", "&Copy Conditions/Effects", "&Paste Conditions/Effects"]
        _select_entries(panel)
        panel.entry_tree.clearSelection()
        assert not window.cut_action.isEnabled() and not window.paste_action.isEnabled()
        _select_entries(panel, ("effect", 1))
        assert window.cut_action.isEnabled() and window.copy_action.isEnabled()

        _focus_on(panel.tree)
        assert _edit_texts(window) == ["Cu&t Triggers", "&Copy Triggers", "&Paste Triggers"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_two_row_entry_selection_pops_up_no_discarded_form_row() -> None:
    """select_entries() of two rows builds the form twice in one gesture. The
    first form's rows still have Qt's queued show pending when they are
    discarded; unhidden, it mapped one as a top-level window that took focus."""
    from PyQt5.QtWidgets import QApplication

    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _focus_on(panel.entry_tree)
        panel.select_entries([("effect", 1), ("effect", 2)])
        QApplication.processEvents()
        strays = [
            w for w in QApplication.topLevelWidgets() if w.isVisible() and w is not window and not window.isAncestorOf(w)
        ]
        assert strays == []
        assert panel.entry_tree.hasFocus()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh137_ctrl_x_c_v_on_the_entry_tree_act_on_conditions_and_effects() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        _focus_on(panel.entry_tree)
        _select_entries(panel, ("effect", 1), ("effect", 2))
        QTest.keyClick(panel.entry_tree, Qt.Key_C, Qt.ControlModifier)
        QApplication.processEvents()
        assert window._entry_clipboard.label == "2 effects"
        assert window._trigger_clipboard is None

        panel.select_trigger(2)
        _focus_on(panel.entry_tree)
        _select_entries(panel, ("effect", 0))
        QTest.keyClick(panel.entry_tree, Qt.Key_V, Qt.ControlModifier)
        QApplication.processEvents()
        assert _types_of(window, 2) == [8, 19, 11, 9], "below the current effect"
        assert len(panel._manager().triggers) == 4, "no trigger was pasted"
        assert panel.entry_tree.hasFocus(), "selecting the two pasted rows keeps the focus"

        _select_entries(panel, ("effect", 3))
        QTest.keyClick(panel.entry_tree, Qt.Key_X, Qt.ControlModifier)
        QApplication.processEvents()
        assert _types_of(window, 2) == [8, 19, 11]
        assert window._entry_clipboard.label == "1 effect"
        assert len(panel._manager().triggers) == 4, "no trigger was cut"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("opener", ["new", "retype"])
def test_gh137_edit_cut_copy_paste_stay_off_while_the_type_picker_is_up(opener: str) -> None:
    """The right-click menus' gate: with New or Type… open on trigger T, a
    click on T's row and Ctrl+X must not cut T out from under the picker."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _active_triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        window.copy_triggers()  # a held block, so a stray Paste would show
        if opener == "retype":
            _select_entries(panel, ("effect", 0))
        panel._request_entry_op(opener)
        QApplication.processEvents()
        assert panel.picker_showing()
        item = panel._item_for_index[3]
        panel.tree.scrollToItem(item)
        rect = panel.tree.visualItemRect(item)
        QTest.mouseClick(panel.tree.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(rect.left() + 8, rect.center().y()))
        QApplication.processEvents()
        assert panel.picker_showing() and panel.tree.hasFocus(), "the click leaves the picker up"
        assert _edit_texts(window) == ["Cu&t Triggers", "&Copy Triggers", "&Paste Triggers"]
        assert not any(a.isEnabled() for a in (window.cut_action, window.copy_action, window.paste_action))

        for name, key in (("X", Qt.Key_X), ("V", Qt.Key_V)):
            QTest.keyClick(panel.tree, key, Qt.ControlModifier)
            QApplication.processEvents()
            assert len(panel._manager().triggers) == 4, f"Ctrl+{name} acted with the picker up"
            assert panel.picker_showing()

        panel._close_picker()
        QApplication.processEvents()
        assert window.cut_action.isEnabled() and window.paste_action.isEnabled(), "closing the picker re-gates them"
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        window.edit_history.mark_saved()
        window.close()


# -- GH #140: the Display Instructions preview ----------------------------------

# The fixture's one Display Instructions effect: trigger 0 "Fixture: setup", effect 0.
_SETUP_TRIGGER = 0


def _instruction_window(**fields):
    """The trigger window with the fixture's Display Instructions effect
    selected, its entry fields first set to `fields` (pre-model, in-test: the
    fixture is not regenerated)."""
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    panel = window.trigger_panel
    panel.select_trigger(_SETUP_TRIGGER)
    entry = panel._manager().triggers[_SETUP_TRIGGER].effects[0]
    assert entry.effect_type == _DISPLAY_INSTRUCTIONS
    for name, value in fields.items():
        setattr(entry, name, value)
    panel.refresh_entries(select=("effect", 0))
    QApplication.processEvents()
    return window


def _preview(panel):
    preview = panel._instruction_preview
    assert preview is not None, "no preview for a single Display Instructions effect"
    return preview


def _live_previews(panel) -> list:
    from descape.instruction_preview import InstructionPreview

    return panel.property_host.findChildren(InstructionPreview)


def _drawn_third(preview) -> int:
    """Which third of the mock screen the box's centre sits in, 0 top to 2 bottom."""
    screen, box = preview.screen, preview.box
    assert screen.height() > 0 and box.height() > 0, "the preview was never laid out"
    return min(2, box.geometry().center().y() * 3 // screen.height())


def test_a_single_display_instructions_effect_gets_a_preview_under_its_form() -> None:
    from PyQt5.QtWidgets import QFormLayout

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        preview = _preview(panel)
        assert _live_previews(panel) == [preview]
        form = panel.property_form
        last = form.itemAt(form.rowCount() - 1, QFormLayout.SpanningRole)
        assert last is not None and last.widget() is preview, "a full-width row below the fields"
        assert "Fixture scenario loaded." in preview.text_label.text()
        assert preview.drawn_position == 0
        assert _drawn_third(preview) == 0
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_no_preview_for_a_trigger_a_condition_another_effect_or_a_multi_select() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_SETUP_TRIGGER)
        QApplication.processEvents()
        assert panel._instruction_preview is None, "the trigger row"
        panel.refresh_entries(select=("condition", 0))
        QApplication.processEvents()
        assert panel._instruction_preview is None, "a condition"
        _select_armour_split_effect(panel, 0)
        QApplication.processEvents()
        assert panel._instruction_preview is None, "a Modify Attribute effect"
        assert _live_previews(panel) == []

        added = _plant_prose_effect(window, "second")
        assert panel._instruction_preview is not None
        _select_entries(panel, ("effect", 0), ("effect", added))
        assert len(panel._rows) > 0, "the two share a form"
        assert panel._instruction_preview is None, "two Display Instructions effects"
        assert _live_previews(panel) == []
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(("stored", "third", "note"), [(0, 0, ""), (1, 1, ""), (2, 2, ""), (-1, 0, "unset, drawn as TOP")])
def test_the_box_sits_in_its_positions_third(stored: int, third: int, note: str) -> None:
    window = _instruction_window(instruction_panel_position=stored)
    try:
        preview = _preview(window.trigger_panel)
        assert preview.drawn_position == third
        assert _drawn_third(preview) == third
        if note:
            assert note in preview.note_label.text()
        else:
            assert "unset" not in preview.note_label.text()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_typing_in_the_message_updates_the_preview_but_writes_nothing() -> None:
    """Observe-only: per keystroke the preview follows, and nothing reaches
    _changed() until the box's own focus-out commit, exactly as before."""
    from descape.message_markup import TEXT_COLORS

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        preview = _preview(panel)
        entry = panel._manager().triggers[_SETUP_TRIGGER].effects[0]
        _row_widget(panel, "message").setPlainText("<BLUE>Scout: <GREY>over here & <fixture_var>")
        html = preview.text_label.text()
        assert "Scout: " in html and "over here &amp; " in html
        assert "color:#{:02x}{:02x}{:02x}".format(*TEXT_COLORS["BLUE"]) in html
        assert "color:#{:02x}{:02x}{:02x}".format(*TEXT_COLORS["GREY"]) in html
        assert "[fixture_var]" in html, "a trigger variable name is a placeholder"
        _row_widget(panel, "sound_name").setText("Play_Technology_Researched")
        assert "Sound: Play_Technology_Researched" in preview.footer_label.text()

        assert entry.message == "Fixture scenario loaded."
        assert entry.sound_name == ""
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
        assert _record_count(window) == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_field_edit_records_only_its_own_undo_step_and_the_preview_follows() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        combo = _row_widget(panel, "instruction_panel_position")
        combo.setCurrentIndex(combo.findData(2))
        QApplication.processEvents()
        assert _record_count(window) == 1, "the position edit, and nothing from the preview"
        preview = _preview(panel)
        assert preview.drawn_position == 2
        assert _drawn_third(preview) == 2

        _row_widget(panel, "play_sound").setChecked(True)
        QApplication.processEvents()
        assert _record_count(window) == 2
        assert "(plays)" in _preview(panel).footer_label.text()
        assert _live_previews(panel) == [panel._instruction_preview]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_reselecting_leaves_no_stale_preview() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        first = _preview(panel)
        _select_armour_split_effect(panel, 0)
        QApplication.processEvents()
        assert panel._instruction_preview is None, "the reference outlived its form"
        assert _live_previews(panel) == []
        assert first.parent() is None, "the old preview is unparented with its row"

        panel.select_trigger(_SETUP_TRIGGER)
        panel.refresh_entries(select=("effect", 0))
        QApplication.processEvents()
        assert _live_previews(panel) == [panel._instruction_preview]
        assert panel._instruction_preview is not first
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    ("play_sound", "words"), [(-1, "(default)"), (0, "(off)"), (1, "(plays)")], ids=["default", "off", "plays"]
)
def test_the_footer_reads_play_sound_from_the_entry(play_sound: int, words: str) -> None:
    """The checkbox shows -1 and 1 alike as checked, so the entry decides."""
    window = _instruction_window(play_sound=play_sound, sound_name="Play_66433", display_time=7)
    try:
        footer = _preview(window.trigger_panel).footer_label.text()
        assert "Sound: Play_66433" in footer and words in footer
        assert "Shows for 7 s" in footer
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(("display_time", "words"), [(0, "Shows for 0 s"), (-1, "Shows for (unset)")])
def test_the_footer_shows_zero_and_unset_display_times(display_time: int, words: str) -> None:
    window = _instruction_window(display_time=display_time)
    try:
        footer = _preview(window.trigger_panel).footer_label.text()
        assert words in footer
        assert "Sound: none" in footer
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_with_no_install_a_string_id_is_not_available_and_the_icon_falls_back() -> None:
    from descape import asset_source

    assert asset_source.get_install_path() is None
    window = _instruction_window(string_id=60014, object_list_unit_id=448)
    try:
        preview = _preview(window.trigger_panel)
        assert "language string #60014 (not available)" in preview.note_label.text()
        assert preview.icon_source in ("sprite", None)
        if preview.icon_source is None:
            assert preview.icon_label.pixmap() is None or preview.icon_label.pixmap().isNull()
    finally:
        window.edit_history.mark_saved()
        window.close()


def _preview_install(tmp_path: Path) -> Path:
    """A fake install: one string, one unit icon (Scout Cavalry's 64) and a Blue override."""
    import json

    from PIL import Image

    install = tmp_path / "install"
    strings = install / "resources" / "en" / "strings" / "key-value"
    strings.mkdir(parents=True)
    (strings / "key-value-strings-utf8.txt").write_text('60014 "<BLUE>From the table"\n', encoding="utf-8")
    units = install / "widgetui" / "textures" / "ingame" / "units"
    units.mkdir(parents=True)
    Image.new("RGBA", (16, 16), (200, 10, 10, 255)).save(units / "064_50730.DDS", format="DDS")
    colors = {"Blue": {"Text": [1, 2, 3, 255]}}
    (install / "widgetui" / "UIColors.json").write_text(json.dumps({"ColorTables": colors}))
    return install


def test_an_install_supplies_the_string_the_icon_and_the_colours(tmp_path: Path) -> None:
    from descape import asset_source

    asset_source.set_install_path_override(_preview_install(tmp_path))
    try:
        window = _instruction_window(string_id=60014, object_list_unit_id=448)
        try:
            preview = _preview(window.trigger_panel)
            assert "From the table" in preview.text_label.text(), "the string replaces the message"
            assert "Fixture scenario loaded." not in preview.text_label.text()
            assert "color:#010203" in preview.text_label.text(), "the install's Blue"
            assert "language string #60014" in preview.note_label.text()
            assert "not available" not in preview.note_label.text()
            assert preview.icon_source == "icon"
            assert not preview.icon_label.pixmap().isNull()
        finally:
            window.edit_history.mark_saved()
            window.close()
    finally:
        asset_source.set_install_path_override(None)


def test_the_icon_frame_takes_the_source_players_colour_or_the_leading_tag_colour() -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.message_markup import TEXT_COLORS

    window = _instruction_window(message="<BLUE>tagged", use_tag_color_for_icon=-1)
    try:
        panel = window.trigger_panel
        colors = window.scenario.player_colors
        assert colors is not None
        assert _preview(panel).tint == tuple(colors[1]), "source player ONE's colour"

        panel.refresh_player_labels(panel._player_labels, None)
        QApplication.processEvents()
        assert _preview(panel).tint is None, "no colours, no tint"

        entry = panel._manager().triggers[_SETUP_TRIGGER].effects[0]
        entry.use_tag_color_for_icon = 1
        panel.refresh_player_labels(panel._player_labels, colors)
        panel.refresh_entries(select=("effect", 0))
        QApplication.processEvents()
        assert _preview(panel).tint == TEXT_COLORS["BLUE"]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.font_sensitive
def test_a_long_message_grows_the_preview_without_overlap_or_widening() -> None:
    """A box taller than a third grows the screen rather than clipping, a long
    unbreakable word cannot widen the pane, and typing re-fits the form."""
    from PyQt5.QtWidgets import QApplication

    from descape.trigger_panel import TriggerPanel

    def check(panel) -> None:
        QApplication.processEvents()
        preview = _preview(panel)
        _assert_form_rows_do_not_overlap(panel.property_form)
        assert preview.minimumSizeHint().width() <= TriggerPanel.MIN_USEFUL_WIDTH
        assert not panel.property_area.horizontalScrollBar().isVisible()
        box, screen = preview.box, preview.screen
        assert box.height() >= box.heightForWidth(box.width()), "the box clips its text"
        assert box.geometry().top() >= 0
        assert box.geometry().bottom() < screen.height()
        assert _drawn_third(preview) == 2
        assert panel.property_host.height() >= panel.property_form.heightForWidth(panel.property_host.width())

    window = _instruction_window(message="<BLUE>" + "W" * 120, instruction_panel_position=2)
    try:
        panel = window.trigger_panel
        check(panel)
        before = _preview(panel).height()
        _row_widget(panel, "message").setPlainText("<GREY>" + "A long line of narration. " * 30)
        check(panel)
        assert _preview(panel).height() > before, "the screen grew with its box"
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_preview_takes_no_focus() -> None:
    """A focusable preview child would pull focus off the message box and
    fire its focus-out commit."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QWidget

    window = _instruction_window()
    try:
        preview = _preview(window.trigger_panel)
        for widget in [preview, *preview.findChildren(QWidget)]:
            assert widget.focusPolicy() == Qt.NoFocus, widget
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #134: a collapsed section header counts as its whole section ------------


def _ab_window(collapse_a: bool = True):
    """The trigger fixture as two sections, --- A --- 0: 1 | --- B --- 2: 3,
    renamed through the widget, with A collapsed by default."""
    window = _triggers_window()
    panel = window.trigger_panel
    _rename(panel, 0, "--- A ---")
    _rename(panel, 2, "--- B ---")
    assert [s.header_index for s in panel._sections] == [0, 2], "fixture assumption"
    panel._item_for_index[0].setExpanded(not collapse_a)
    return window


def _display_names(window) -> list[str]:
    manager = window.trigger_panel._manager()
    return [manager.triggers[i].name for i in manager.trigger_display_order]


def _section_names(window) -> list[tuple[str, list[str]]]:
    panel = window.trigger_panel
    triggers = panel._manager().triggers
    return [(triggers[s.header_index].name, [triggers[m].name for m in s.member_indices]) for s in panel._sections]


def test_gh134_duplicating_a_collapsed_section_copies_it_whole_after_itself() -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        assert not panel._item_for_index[0].isExpanded(), "selecting the header leaves it collapsed"
        records = _record_count(window)
        panel.trigger_copy_button.click()
        assert _display_names(window) == [
            "--- A ---", "Fixture: armour split",
            "--- A (copy) ---", "Fixture: armour split (copy)",
            "--- B ---", "Fixture: variable",
        ]
        assert _section_names(window) == [
            ("--- A ---", ["Fixture: armour split"]),
            ("--- A (copy) ---", ["Fixture: armour split (copy)"]),
            ("--- B ---", ["Fixture: variable"]),
        ]
        assert _record_count(window) == records + 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_deleting_a_collapsed_section_removes_it_whole_in_one_record() -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        records = _record_count(window)
        panel.trigger_delete_button.click()
        assert _display_names(window) == ["--- B ---", "Fixture: variable"]
        assert _record_count(window) == records + 1
        window.undo()
        assert _display_names(window) == ["--- A ---", "Fixture: armour split", "--- B ---", "Fixture: variable"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_cutting_a_collapsed_section_cuts_it_whole() -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        window.cut_triggers()
        assert _display_names(window) == ["--- B ---", "Fixture: variable"]
        assert [t.name for t in window._trigger_clipboard.triggers] == ["--- A ---", "Fixture: armour split"]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("collapse_b", [True, False], ids=["B collapsed", "B expanded"])
def test_gh134_a_copied_collapsed_section_pastes_as_its_own_section_after_b(collapse_b: bool) -> None:
    """Collapsed, the panel resolves the anchor to B's last member; expanded,
    the window's divider-block rule does, so B never loses its member."""
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        panel.trigger_clipboard_copy_button.click()
        assert [t.name for t in window._trigger_clipboard.triggers] == ["--- A ---", "Fixture: armour split"]
        panel._item_for_index[2].setExpanded(not collapse_b)
        panel.select_trigger(2)
        panel.trigger_paste_button.click()
        assert _section_names(window) == [
            ("--- A ---", ["Fixture: armour split"]),
            ("--- B ---", ["Fixture: variable"]),
            ("--- A (copy) ---", ["Fixture: armour split (copy)"]),
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("entry", ["edit", "button", "menu"])
def test_gh134_a_plain_paste_onto_a_collapsed_header_lands_at_its_section_end(entry: str) -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        window.copy_triggers()
        panel.select_trigger(0)
        if entry == "edit":
            window.paste_triggers()
        elif entry == "button":
            panel.trigger_paste_button.click()
        else:
            _action(panel.trigger_context_menu(panel._item_for_index[0]), "Paste").trigger()
        assert _display_names(window) == [
            "--- A ---", "Fixture: armour split", "Fixture: variable (copy)", "--- B ---", "Fixture: variable",
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_duplicating_an_expanded_header_alone_leaves_its_members() -> None:
    """The split guard: the copy is an empty section after A, not a header
    that takes A's member."""
    window = _ab_window(collapse_a=False)
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        panel.trigger_copy_button.click()
        assert _section_names(window) == [
            ("--- A ---", ["Fixture: armour split"]),
            ("--- A (copy) ---", []),
            ("--- B ---", ["Fixture: variable"]),
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_scattered_duplicates_each_land_after_their_source() -> None:
    window = _ab_window(collapse_a=False)
    try:
        panel = window.trigger_panel
        _ctrl_select(panel, 1, 3)
        panel.trigger_copy_button.click()
        assert _display_names(window) == [
            "--- A ---", "Fixture: armour split", "Fixture: armour split (copy)",
            "--- B ---", "Fixture: variable", "Fixture: variable (copy)",
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh134_move_up_and_down_stay_header_only_on_a_collapsed_section() -> None:
    window = _ab_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        panel.trigger_move_down_button.click()
        assert _order(window) == [1, 0, 2, 3]
        window.undo()
        panel._item_for_index[2].setExpanded(False)
        panel.select_trigger(2)
        panel.trigger_move_up_button.click()
        assert _order(window) == [0, 2, 1, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #133: Section Up / Section Down move a whole section -------------------


@pytest.mark.parametrize("collapse_a", [True, False], ids=["A collapsed", "A expanded"])
def test_gh133_section_down_swaps_two_sections_in_one_record(collapse_a: bool) -> None:
    """The two renames push records of their own, so the pin is that the top
    record is the section move and one undo restores the order with the
    divider names still in place."""
    from descape.edit_history import TriggerDiffRecord

    window = _ab_window(collapse_a=collapse_a)
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        assert panel.section_down_button.isEnabled()
        assert not panel.section_up_button.isEnabled()
        records = _record_count(window)
        panel.section_down_button.click()
        assert _order(window) == [2, 3, 0, 1]
        assert _section_names(window) == [
            ("--- B ---", ["Fixture: variable"]),
            ("--- A ---", ["Fixture: armour split"]),
        ]
        assert _record_count(window) == records + 1
        record = window.edit_history.peek_undo()
        assert isinstance(record, TriggerDiffRecord) and record.label == 'Move section "--- A ---" down'
        assert record.touched == []
        assert panel.selected_trigger_indices() == [0], "the selection follows the moved section"
        assert panel.section_up_button.isEnabled() and not panel.section_down_button.isEnabled()

        window.undo()
        assert _order(window) == [0, 1, 2, 3]
        assert _names(window)[0] == "--- A ---" and _names(window)[2] == "--- B ---"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh133_a_member_moves_its_section_and_move_up_still_moves_a_divider_alone() -> None:
    window = _ab_window(collapse_a=False)
    try:
        panel = window.trigger_panel
        panel.select_trigger(3)
        panel.section_up_button.click()
        assert _order(window) == [2, 3, 0, 1]
        assert window.edit_history.peek_undo().label == 'Move section "--- B ---" up'
        window.undo()
        panel.select_trigger(2)
        panel.trigger_move_up_button.click()
        assert _order(window) == [0, 2, 1, 3], "▲ still re-partitions: B takes A's member"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh133_the_leading_run_neither_moves_nor_is_swapped_past() -> None:
    """(before) 0 | --- B --- 1: 2, 3. B's Section Up would hand trigger 0
    to B, so it is off and says why; nothing is below B either."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _rename(panel, 1, "--- B ---")
        panel.select_trigger(1)
        assert not panel.section_up_button.isEnabled()
        assert panel.section_up_button.toolTip().endswith(" (no section above this one)")
        assert not panel.section_down_button.isEnabled()
        assert panel.section_down_button.toolTip().endswith(" (no section below this one)")
        panel.select_trigger(0)
        for button in (panel.section_up_button, panel.section_down_button):
            assert not button.isEnabled()
            assert "before the first section" in button.toolTip()
        records = _record_count(window)
        window.trigger_structural_edit("section_up", [1])
        window.trigger_structural_edit("section_down", [0])
        assert _record_count(window) == records, "a refused move records nothing"
        assert _order(window) == [0, 1, 2, 3]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh133_section_moves_are_gated_like_move_up_and_down() -> None:
    window = _ab_window(collapse_a=False)
    try:
        panel = window.trigger_panel
        panel.select_trigger(1)
        assert panel.section_down_button.isEnabled()

        panel.filter_edit.setText("a")
        panel.select_trigger(1)
        assert not panel.section_down_button.isEnabled()
        assert "clear the filter" in panel.section_down_button.toolTip()
        panel.filter_edit.clear()

        _rename(panel, 3, "[t] tagged")
        panel.set_tag_filter("t")
        panel.select_trigger(3)
        assert not panel.section_up_button.isEnabled()
        assert "clear the filter or tag" in panel.section_up_button.toolTip()
        panel.set_tag_filter(None)

        panel.sort_combo.setCurrentIndex(1)
        panel.select_trigger(1)
        assert not panel.section_down_button.isEnabled()
        assert "Display order" in panel.section_down_button.toolTip()
        panel.sort_combo.setCurrentIndex(0)
        panel.select_trigger(1)
        assert panel.section_down_button.isEnabled()

        _ctrl_select(panel, 1, 3)
        assert not panel.section_down_button.isEnabled() and not panel.section_up_button.isEnabled()
        assert "one section" in panel.section_down_button.toolTip()
        records = _record_count(window)
        window.trigger_structural_edit("section_down", [1, 3])
        assert _record_count(window) == records, "a selection spanning sections records nothing"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_gh133_a_flat_file_offers_no_section_move() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(1)
        assert not panel.section_up_button.isEnabled() and not panel.section_down_button.isEnabled()
        assert "no sections" in panel.section_up_button.toolTip()
    finally:
        window.edit_history.mark_saved()
        window.close()


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
    from PyQt5.QtCore import Qt

    brush = item.data(col, Qt.ForegroundRole)
    return None if brush is None else brush.color().name()


def _icon_image(item, col: int):
    return item.icon(col).pixmap(16).toImage()


def _expected_icon(status, hex_str: str):
    from descape.trigger_panel import status_icon

    return status_icon(status, hex_str).pixmap(16).toImage()


def _effect_type_id(panel, name: str) -> int:
    return next(i for i, d in panel._vocabulary.effects.items() if d.name == name)


def test_the_panels_own_status_style_defaults_are_the_settings_defaults() -> None:
    from descape import settings
    from descape.trigger_panel import StatusStyle

    style = StatusStyle()
    assert style.marker == settings.TRIGGER_STATUS_MARKER_DEFAULT
    assert style.ok == settings.get_default_trigger_status_color("ok")
    assert style.problem == settings.get_default_trigger_status_color("problem")
    assert style.color_ok_rows is True


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
        col = panel._COL_NAME
        for index, status, hex_str in ((0, Status.OK, _OK_HEX), (_BROKEN_TRIGGER, Status.PROBLEM, _PROBLEM_HEX)):
            row = panel._item_for_index[index]
            assert _brush_hex(row, col) == (hex_str if brushed else None), (marker, index)
            assert row.icon(col).isNull() is not iconed, (marker, index)
            if iconed:
                assert _icon_image(row, col) == _expected_icon(status, hex_str), (marker, index)
        # The tooltip is not a marker: every mode, "off" included, explains a problem row.
        assert panel._item_for_index[_BROKEN_TRIGGER].toolTip(col)
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
        col = panel._COL_NAME
        panel.select_trigger(_BROKEN_TRIGGER)
        style = panel._status_style
        for ok_row, ok_col in ((panel._item_for_index[0], col), (panel._entry_item_for("effect", 0), 0)):
            assert _brush_hex(ok_row, ok_col) is None, marker
            assert ok_row.icon(ok_col).isNull(), marker
        for problem_row, problem_col in ((panel._item_for_index[_BROKEN_TRIGGER], col),
                                         (panel._entry_item_for("effect", 6), 0)):
            assert _brush_hex(problem_row, problem_col) == (_PROBLEM_HEX if style.colours else None), marker
            assert problem_row.icon(problem_col).isNull() is not style.icons, marker
            if style.icons:
                assert _icon_image(problem_row, problem_col) == _expected_icon(Status.PROBLEM, _PROBLEM_HEX)
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
    from descape.trigger_status import Status
    from descape.viewer import SettingsDialog

    window = _triggers_window()
    dialog = SettingsDialog(window)
    try:
        panel = window.trigger_panel
        col = panel._COL_NAME
        broken, ok = panel._item_for_index[_BROKEN_TRIGGER], panel._item_for_index[0]
        panel.select_trigger(_BROKEN_TRIGGER)

        combo = dialog.trigger_status_marker_combo
        combo.setCurrentIndex(combo.findData("icon"))
        assert settings.get_trigger_status_marker() == "icon"
        assert _brush_hex(broken, col) is None and not broken.icon(col).isNull()
        assert not panel._entry_item_for("effect", 6).icon(0).isNull()

        combo.setCurrentIndex(combo.findData("both"))
        assert dialog.trigger_status_ok_rows_check.text() == "Mark triggers that pass"
        dialog.trigger_status_ok_rows_check.setChecked(False)
        assert _brush_hex(ok, col) is None and ok.icon(col).isNull()
        assert _icon_image(broken, col) == _expected_icon(Status.PROBLEM, _PROBLEM_HEX)
        dialog.trigger_status_ok_rows_check.setChecked(True)
        assert _icon_image(ok, col) == _expected_icon(Status.OK, _OK_HEX)

        dialog._apply_trigger_status_color("problem", "#123456")
        assert settings.get_trigger_status_color("problem") == "#123456"
        assert _brush_hex(broken, col) == "#123456"
        assert _icon_image(broken, col) == _expected_icon(Status.PROBLEM, "#123456")
        assert _brush_hex(panel._entry_item_for("effect", 6), 0) == "#123456"
    finally:
        dialog.close()
        window.close()


def test_a_divider_row_stays_unstyled_while_its_entries_are_styled() -> None:
    from descape import settings

    settings.set_trigger_status_marker("both")
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        col = panel._COL_NAME
        panel.select_trigger(1)
        name = _row_widget(panel, "name")
        name.setText("--- Section ---")
        name.editingFinished.emit()  # crosses is_divider: a full repopulate

        divider = panel._item_for_index[1]
        assert _brush_hex(divider, col) is None
        assert divider.icon(col).isNull()
        assert divider.toolTip(col) == ""
        panel.select_trigger(1)
        assert _brush_hex(panel.entry_tree.topLevelItem(0), 0) is None
        assert _brush_hex(panel._entry_item_for("effect", 0), 0) == _OK_HEX
    finally:
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


def test_a_live_marker_change_sizes_the_name_columns_like_a_fresh_populate() -> None:
    """An icon widens the name column. A live restyle batches its setData calls
    by switching the trigger tree's columns off ResizeToContents, so they must
    come back, and both trees must end up as wide as a repopulate makes them."""
    from PyQt5.QtWidgets import QApplication, QHeaderView

    from descape import settings

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_BROKEN_TRIGGER)
        QApplication.processEvents()
        before = (panel.tree.header().sectionSize(panel._COL_NAME), panel.entry_tree.columnWidth(0))

        settings.set_trigger_status_marker("icon")
        panel.apply_status_style()
        QApplication.processEvents()
        header = panel.tree.header()
        assert all(header.sectionResizeMode(c) == QHeaderView.ResizeToContents for c in range(header.count()))
        live = (header.sectionSize(panel._COL_NAME), panel.entry_tree.columnWidth(0))

        window._show_triggers()
        QApplication.processEvents()
        fresh = (header.sectionSize(panel._COL_NAME), panel.entry_tree.columnWidth(0))
        assert live == fresh
        assert live[0] > before[0] and live[1] > before[1], "the icon did not widen the name columns"
    finally:
        window.close()


def _count_evaluations(monkeypatch) -> list:
    """Every trigger object evaluate_trigger() is called on, in call order."""
    from descape import trigger_status

    real = trigger_status.evaluate_trigger
    calls: list = []

    def counting(trigger, context):
        calls.append(trigger)
        return real(trigger, context)

    monkeypatch.setattr(trigger_status, "evaluate_trigger", counting)
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
        assert panel.status_of(_BROKEN_TRIGGER).trigger.status is Status.NONE
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
        assert panel.status_of(_BROKEN_TRIGGER).trigger.status is Status.NONE
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
    import time

    from PyQt5.QtWidgets import QApplication

    from descape import settings

    path = Path(__file__).resolve().parent.parent / "examples" / "old-allies-final-v2.aoe2scenario"
    if not path.is_file():
        pytest.skip(f"missing corpus file: {path}")
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
