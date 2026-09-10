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

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"
TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def _window():
    """A shown, fixed-size offscreen ViewerWindow.

    show() + processEvents() is load-bearing for anything reading the
    splitter: QSplitter.setSizes() is renormalized against the widget's real
    geometry, so on an unshown window every size assertion tests Qt's layout
    fallback rather than the code under test.
    """
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.resize(1500, 900)
    window.show()
    QApplication.processEvents()
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
            panel._changed(spec, "condition", 0, 4242)
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
        assert panel.tree.topLevelItem(0).text(1) == "Renamed in the panel", (
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
        widget = panel._build_document_reference_widget(spec, "effect", 0, UNSET, True)
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

        window.trigger_structural_edit("delete", 0)  # delete "Fixture: setup" (id 0)

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
        panel._changed(spec, "condition", 0, 4242)
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
            window.trigger_structural_edit("new", -1)
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
        panel.entry_tree.setCurrentItem(_group(panel, "effect"))
        assert not panel.entry_copy_button.isEnabled()
        assert not panel.entry_delete_button.isEnabled()

        _select_first_condition(panel)
        assert panel.entry_copy_button.isEnabled()
        assert panel.entry_delete_button.isEnabled()

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
        ):
            assert not button.isEnabled(), f"{button.text()} is live on a read-only file"

        # And the gate holds below the buttons too, not just on them.
        panel._request_trigger_op("new")
        panel._request_entry_op("new")
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

        window.trigger_structural_edit("move_down", 1)  # row 1 -> target 2, both valid rows

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

        window.trigger_structural_edit("move_up", moved_index)

        assert _tree_order(panel)[0] == moved_index
        after_items = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        assert set(id(i) for i in before_items) == set(id(i) for i in after_items), (
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

        window.trigger_structural_edit("move_up", moved_index)
        assert _tree_order(panel) != original
        window.trigger_structural_edit("move_down", moved_index)
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
        window.trigger_structural_edit("move_up", moved_index)
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
    source = inspect.getsource(ViewerWindow.trigger_structural_edit)
    assert ".reorder_triggers(" not in source
    assert ".move_triggers(" not in source


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
    nothing to scroll to. Two columns now: ID is asserted with >=, not ==,
    since ResizeToContents sizes a section to the header's own hint too, and
    "ID" is wider than a one-digit fixture value -- sizeHintForColumn() looks
    at row content only. Trigger keeps strict equality, unchanged from before
    the ID column existed."""
    window = _triggers_window()
    try:
        tree = window.trigger_panel.tree
        assert tree.columnWidth(0) >= tree.sizeHintForColumn(0)
        assert tree.columnWidth(1) == tree.sizeHintForColumn(1)
        if tree.sizeHintForColumn(0) + tree.sizeHintForColumn(1) <= tree.viewport().width():
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
            assert int(item.text(0)) == index

        panel = window.trigger_panel
        needle = tree.topLevelItem(0).text(0)
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


def test_a_long_form_scrolls_instead_of_crushing_its_rows() -> None:
    """Found by screenshot, and invisible to every other measurement: host
    width, scrollbar state and field widths all read correct while each row was
    squeezed to 6 px and drew over the next one.

    The cause is that a wrapped row's height depends on its width, and a
    QScrollArea sizes its widget from sizeHint(), computed as if nothing
    wrapped -- so the form was handed the unwrapped height and overflowed
    inside it. Asking the layout for heightForWidth, *after* activating it, is
    what makes the vertical scrollbar appear instead.
    """
    from PyQt5.QtWidgets import QFormLayout

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_an_effect(panel)
        form = panel.property_form
        assert form.rowCount() >= 8, "this fixture no longer exercises a long form"

        previous_bottom = None
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.LabelRole)
            field = form.itemAt(row, QFormLayout.FieldRole)
            if not (label and label.widget() and field and field.widget()):
                continue
            label_geometry = label.widget().geometry()
            field_geometry = field.widget().geometry()
            assert field.widget().height() >= field.widget().minimumSizeHint().height(), (
                f"{label.widget().text()}'s editor is squeezed below its own minimum height"
            )
            if previous_bottom is not None:
                assert label_geometry.top() >= previous_bottom, (
                    f"{label.widget().text()} draws over the row above it"
                )
            previous_bottom = max(label_geometry.bottom(), field_geometry.bottom())

        assert panel.property_host.height() > panel.property_area.viewport().height(), (
            "a form taller than its pane must make the host taller, not compress the rows"
        )
        assert panel.property_area.verticalScrollBar().isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


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
                    for spec, _kind, _index, widget in panel._rows:
                        checked += 1
                        assert (
                            widget.minimumSizeHint().width() <= TriggerPanel.MIN_USEFUL_WIDTH
                        ), (
                            f"{spec.name}'s editor demands "
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
        panel.filter_edit.setText(real.text(1))
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
            header_matches = top.data(0, Qt.UserRole) is not None and top.data(0, panel._TAG_ROLE) == wanted
            # A visible row must owe its visibility to a real match -- its own
            # tag (flat rows and real section headers), or a visible child's.
            assert header_matches or visible_children
            for child in visible_children:
                assert child.data(0, panel._TAG_ROLE) == wanted
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

        assert first.text(0) == "", "the synthetic header is not a trigger and has no id to show"
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
