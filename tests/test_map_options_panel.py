"""Coverage for Map Options mode and MapOptionsPanel.

The panel is editable as of step 3, which changes what the tests here are
worth. `test_browsing_every_row_reports_no_edit` was explicitly *not*
load-bearing while every widget was disabled -- a disabled widget emits no
signal, so it passed with `_changed()`'s guards deleted. It is now the real
gate it was written to be, driven against enabled widgets, and
`test_browsing_every_row_saves_byte_identically` is the acceptance gate above
it: a phantom record would both dirty the document and reserialise a Triggers
section the user only looked at.

Which rows are editable is two independent gates, not one flag, and the window
owns both: options_model verifies the byte-patched scalars and says nothing
about the trigger execution-order row, while trigger_write_supported says
nothing about the scalars. The panel is handed the resolved set of field ids.

Same offscreen-ViewerWindow technique as tests/test_trigger_panel.py.
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
    """A shown, fixed-size offscreen ViewerWindow -- show() + processEvents()
    is load-bearing for the splitter assertions, see test_trigger_panel.py's
    own copy of this helper for why."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.resize(1500, 900)
    window.show()
    QApplication.processEvents()
    return window


def _close(window) -> None:
    # mark_saved() first: closeEvent -> _confirm_discard_changes() pops a modal
    # on a dirty document, which blocks forever offscreen (conftest.py:91-96).
    window.edit_history.mark_saved()
    window.close()


def _shown_value(panel, spec) -> int:
    """The raw value a row is actually displaying, whichever widget it built.

    A QLabel row is the out-of-range case: the file holds a value no editor on
    that row could hold, so it is shown as stored.
    """
    from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QSpinBox

    widget = panel.widget_for(spec.field_id)
    if isinstance(widget, QCheckBox):
        return int(widget.isChecked())
    if isinstance(widget, QComboBox):
        return widget.currentData()
    if isinstance(widget, QDoubleSpinBox):
        return int(round(widget.value() * spec.scale))
    if isinstance(widget, QLabel):
        return int(widget.text())
    assert isinstance(widget, QSpinBox), f"{spec.field_id} built a {type(widget).__name__}"
    return widget.value()


def _is_read_only(widget) -> bool:
    """A QLabel row is read-only by construction and stays enabled on purpose
    (see MapOptionsPanel._out_of_range_label); every other kind is gated by
    isEnabled()."""
    from PyQt5.QtWidgets import QLabel

    return isinstance(widget, QLabel) or not widget.isEnabled()


def _options_window(path=BLANK_FIXTURE):
    window = _window()
    window.load_scenario(path)
    window.mode_combo.setCurrentText("Map Options")
    return window


# --- the mode ---------------------------------------------------------------


def test_map_options_mode_swaps_the_left_panel_and_gates_edit_tools() -> None:
    from descape.viewer import _LEFT_PAGE_MAP_OPTIONS

    window = _window()
    try:
        assert window.left_stack.currentIndex() == 0
        window.mode_combo.setCurrentText("Map Options")
        assert window.mode == "map_options"
        assert window.left_stack.currentIndex() == _LEFT_PAGE_MAP_OPTIONS == 3
        # Same containment View and Triggers have: no edit tool is reachable,
        # and the active tool falls back to Pan rather than staying checked.
        assert not window.draw_action.isEnabled()
        assert not window.elevation_action.isEnabled()
        assert window.pan_action.isChecked()

        window.mode_combo.setCurrentText("View")
        assert window.left_stack.currentIndex() == 0
    finally:
        _close(window)


def test_the_mode_id_is_a_single_word_and_the_labels_stay_spelled_out() -> None:
    """`self.mode = mode_text.lower()` would make the label "Map Options" into
    the id "map options", with a space, while every gate in viewer.py compares
    against a single lowercase word. The status line and the log take the
    other direction: they read the combo's label, so they must not print the
    id back with an underscore."""
    window = _window()
    try:
        window.mode_combo.setCurrentText("Map Options")
        assert window.mode == "map_options"
        assert "Mode: Map Options" in window.mode_status_label.text()
        assert "Map Options" in window.status_log.toPlainText().splitlines()[-1]
        assert "Map_options" not in window.status_log.toPlainText()
    finally:
        _close(window)


def test_the_four_existing_modes_still_read_the_way_they_always_did() -> None:
    """_update_mode_status() switched from self.mode.capitalize() to the
    combo's own text. Byte-identical for every single-word mode -- pinned so a
    future relabel cannot quietly change the status bar."""
    window = _window()
    try:
        for label in ("View", "Terrain", "Units", "Triggers"):
            window.mode_combo.setCurrentText(label)
            assert f"Mode: {label}" in window.mode_status_label.text()
    finally:
        _close(window)


def test_entering_map_options_widens_a_too_narrow_pane() -> None:
    from descape.map_options_panel import MapOptionsPanel

    window = _window()
    try:
        window.content_splitter.setSizes([200, 1000])
        window.mode_combo.setCurrentText("Map Options")
        assert window.content_splitter.sizes()[0] >= MapOptionsPanel.MIN_USEFUL_WIDTH
    finally:
        _close(window)


def test_widening_uses_this_panels_own_threshold_not_the_trigger_panels() -> None:
    """_widen_left_column() reads its threshold twice -- the guard and the
    min(). Both must be the parameter, so this asserts an upper bound as well
    as a lower one: a min() left reading TriggerPanel.MIN_USEFUL_WIDTH would
    still satisfy ">= 300" while quietly widening to 340.

    Only the min() is observable, as it happens. Left at the trigger panel's
    constant, the *guard* alone changes nothing, because the min() then caps
    the result below the width that would have been let through -- checked by
    mutation rather than assumed. The guard is parameterised anyway, since
    which of the two is load-bearing is an accident of 300 < 340.
    """
    from descape.map_options_panel import MapOptionsPanel
    from descape.trigger_panel import TriggerPanel

    assert MapOptionsPanel.MIN_USEFUL_WIDTH < TriggerPanel.MIN_USEFUL_WIDTH
    window = _window()
    try:
        window.content_splitter.setSizes([200, 1000])
        window.mode_combo.setCurrentText("Map Options")
        left = window.content_splitter.sizes()[0]
        assert left >= MapOptionsPanel.MIN_USEFUL_WIDTH
        assert left < TriggerPanel.MIN_USEFUL_WIDTH, (
            f"widened to {left} px -- that is the trigger panel's threshold, "
            "not this one's"
        )
    finally:
        _close(window)


# --- populate / clear symmetry ----------------------------------------------


def test_panel_populates_on_entering_the_mode() -> None:
    window = _options_window()
    try:
        panel = window.map_options_panel
        assert panel._specs, "no rows built"
        assert set(panel.current_values()) == {s.field_id for s in panel._specs}
        assert "setting" in panel.status.text()
    finally:
        _close(window)


def test_opening_a_map_outside_the_mode_does_not_populate() -> None:
    """Same on-demand contract Triggers mode has: the exec-order row needs the
    Triggers section parsed, and opening a map for terrain work must not pay
    for a 1.17 MB parse nobody asked to see."""
    window = _window()
    try:
        window.load_scenario(BLANK_FIXTURE)
        assert window.map_options_panel._specs == ()
        assert window.map_options_panel.status.text() == "No map open."
    finally:
        _close(window)


def test_closing_a_map_clears_the_panel() -> None:
    window = _options_window()
    try:
        assert window.map_options_panel._specs
        window.edit_history.mark_saved()
        window.close_scenario()
        assert window.map_options_panel._specs == ()
        assert window.map_options_panel.current_values() == {}
        assert window.map_options_panel.status.text() == "No map open."
    finally:
        _close(window)


def test_opening_a_second_map_while_in_the_mode_repopulates() -> None:
    """load_scenario's populate/clear is one if/else per panel, never a chain.

    Chained onto the trigger panel's, the map-options arm would be reached
    only when the mode is not "triggers", and the trigger panel's own clear
    would be the branch that disappears -- leaving it holding a document that
    is no longer open. So this populates the trigger panel first, which is the
    only state where the two shapes differ.
    """
    window = _window()
    try:
        window.load_scenario(BLANK_FIXTURE)
        window.mode_combo.setCurrentText("Triggers")
        assert window.trigger_panel._loaded is window.scenario
        window.mode_combo.setCurrentText("Map Options")

        window.edit_history.mark_saved()
        window.load_scenario(TRIGGER_FIXTURE)
        assert window.map_options_panel._loaded is window.scenario
        # Not this mode's page: cleared, not left pointing at the old document.
        assert window.trigger_panel._loaded is None
    finally:
        _close(window)


def test_leaving_and_re_entering_the_mode_rebuilds_the_form() -> None:
    window = _options_window()
    try:
        first = window.map_options_panel.widget_for("collide_and_correct")
        window.mode_combo.setCurrentText("View")
        window.mode_combo.setCurrentText("Map Options")
        second = window.map_options_panel.widget_for("collide_and_correct")
        assert second is not None and second is not first, "the host was not rebuilt"
    finally:
        _close(window)


# --- the rows themselves ----------------------------------------------------


def test_the_exec_order_row_is_present_once_triggers_are_parsed() -> None:
    """option_fields.specs_for() reads presence off the loaded sections, and
    "Triggers" is absent from them until parse_triggers() has run -- so this
    row exists only because _show_map_options() parses first. Without that the
    panel would silently be one row short of what the plan specifies."""
    window = _options_window(TRIGGER_FIXTURE)
    try:
        panel = window.map_options_panel
        assert "legacy_exec_order" in panel.current_values()
        widget = panel.widget_for("legacy_exec_order")
        assert widget is not None
        assert [widget.itemText(i) for i in range(widget.count())] == [
            "Display order",
            "Legacy - trigger ID order",
        ]
    finally:
        _close(window)


def test_every_row_shows_the_files_own_value() -> None:
    """The panel is a view over the parsed retrievers, so every widget must
    read back the value option_fields.current_value() reports for it -- a
    populate that quietly defaulted a field would be invisible until step 3
    saved that default over the user's data."""
    from descape import option_fields

    window = _options_window(TRIGGER_FIXTURE)
    try:
        panel = window.map_options_panel
        for spec in panel._specs:
            raw = int(option_fields.current_value(window.scenario, spec))
            shown = _shown_value(panel, spec)
            assert shown == raw, f"{spec.field_id} shows {shown}, file holds {raw}"
        # 11, not 15: the four Diplomacy-group scalars moved to
        # DiplomacyPanel (step 5) and no longer count here.
        assert len(panel._specs) == 11, f"{len(panel._specs)} rows -- the spec list changed"
    finally:
        _close(window)


def test_a_scaled_field_shows_file_units_divided_by_its_scale() -> None:
    """victory_years (a count of 10ths of a year) is the panel's one live
    scaled spec. The direction of the division and the floor() on the way
    back were both settled by measurement against the corpus and the
    library before it had a real field to serve, and this is the test that
    keeps that live now.

    victory_years stores 9000 on the blank template, so scale 10 makes the
    row read 900.0. Two halves, because either alone is vacuous: the first
    pins the direction (a scale applied the wrong way round would show 90),
    the second pins the widget kind against a value with a real tenth, which
    an integer spinbox would truncate.
    """
    from PyQt5.QtWidgets import QDoubleSpinBox

    from descape.scenario_io import load_map_and_units
    from descape.map_options_panel import MapOptionsPanel

    conftest.ensure_qapp()
    loaded = load_map_and_units(BLANK_FIXTURE)
    parsed = loaded._scenario.sections["GlobalVictory"].retriever_map[
        "time_for_timed_game_in_10ths_of_a_year"
    ]
    original = int(parsed.data)
    assert original == 9000, f"fixture stores {original} -- this test's numbers assume 9000"

    panel = MapOptionsPanel()
    try:
        panel.show_scenario(loaded)
        widget = panel.widget_for("victory_years")
        assert isinstance(widget, QDoubleSpinBox)
        assert widget.decimals() == 1
        assert widget.value() == pytest.approx(900.0), "raw 9000 at scale 10 must read as 900.0"

        panel._values = {**panel._values, "victory_years": 1250}
        panel._rebuild_host()
        panel._build_groups()
        assert panel.widget_for("victory_years").value() == pytest.approx(125.0), (
            "the tenth was truncated"
        )
    finally:
        parsed.data = original
        panel.deleteLater()


def test_a_scaled_edit_reports_file_units_not_display_units() -> None:
    """The other half of the scale contract: what goes back out. Mirrors
    OptionManager.victory_years' own `floor(value * 10)` setter, so a value
    round-trips through the library's accessor unchanged."""
    from descape import option_fields
    from descape.scenario_io import load_map_and_units
    from descape.map_options_panel import MapOptionsPanel

    conftest.ensure_qapp()
    loaded = load_map_and_units(BLANK_FIXTURE)
    reported = []
    panel = MapOptionsPanel(on_option_field=lambda s, v: reported.append((s.field_id, v)))
    try:
        panel.show_scenario(
            loaded,
            editable_fields=[spec.field_id for spec in option_fields.specs_for(loaded)],
        )
        widget = panel.widget_for("victory_years")
        widget.setValue(130.3)
        assert reported == [("victory_years", 1303)], (
            "display units reached the callback instead of file units"
        )
        assert panel.current_values()["victory_years"] == 1303
    finally:
        panel.deleteLater()


def test_a_value_its_editor_could_not_show_truthfully_is_shown_as_stored() -> None:
    """Not cosmetic: this value rendered as an ordinary ticked checkbox that
    step 3 would have saved as 1, destroying the stored byte.

    villager_force_drop carries 90/119/167/255 on the six 1.41 corpus files
    -- but no shipped fixture does (both are 1.58), so end-to-end this only
    runs in the corpus tier, which test_map_options_panel_populates_every_
    corpus_file covers. The past-int32-spinbox counterpart of this test
    (max_number_of_teams overridden with 0xFFFFFFFF) moved to
    test_diplomacy_panel.py with the rest of the Teams group (step 5); the
    sentinel case (required_score_for_score_victory's 0xFFFFFFFF on five
    corpus files) is covered end-to-end by
    test_a_stored_sentinel_score_is_shown_as_stored_across_the_joan_files
    below.
    """
    field_id, stored = "villager_force_drop", 90
    from PyQt5.QtWidgets import QLabel

    from descape.scenario_io import load_map_and_units
    from descape.map_options_panel import MapOptionsPanel

    conftest.ensure_qapp()
    loaded = load_map_and_units(BLANK_FIXTURE)
    panel = MapOptionsPanel()
    try:
        panel.show_scenario(loaded, values={field_id: stored})
        widget = panel.widget_for(field_id)
        assert isinstance(widget, QLabel), f"built a {type(widget).__name__}, not a label"
        assert widget.text() == str(stored), "the stored value was not shown verbatim"
        assert "editable range" in widget.toolTip()
        # Still reported onward as stored, so step 3's model sees what the row
        # holds rather than a clamped or coerced stand-in.
        assert panel.current_values()[field_id] == stored
        assert widget.minimumSizeHint().width() <= MapOptionsPanel.MIN_USEFUL_WIDTH
        # Signalled on screen too, not only in the row's tooltip: a tooltip is
        # not discoverable from a bare number sitting where a checkbox or a
        # spinbox was expected. The visible half is the count -- the sentence
        # explaining it moved to the status label's own tooltip when spelling
        # every note out inline started pushing the form below the fold (see
        # test_the_status_line_stays_short_enough_to_leave_the_form_visible).
        # The count is what has to stay visible; the prose is a hover away on
        # either the status or the row itself.
        assert "1 shown as stored" in panel.status.text()
        assert (
            "1 setting holds a value no editor could show truthfully"
            in panel.status.toolTip()
        )

        # The file's own in-range value on the same row still gets a real
        # editor, so this is a per-value decision rather than a dead field.
        panel.show_scenario(loaded)
        assert not isinstance(panel.widget_for(field_id), QLabel)
        assert "shown as stored" not in panel.status.text()
        assert "no editor could show truthfully" not in panel.status.toolTip()
    finally:
        panel.deleteLater()


_JOAN_SENTINEL_FILES = [
    "2_Joan_coop_1_v0_13.aoe2scenario",
    "2_Joan_coop_2_v0_15.aoe2scenario",
    "2_Joan_coop_3_v0_14.aoe2scenario",
    "2_Joan_coop_4_v0_13.aoe2scenario",
    "2_Joan_coop_6_v0_14.aoe2scenario",
]
# Named explicitly rather than globbed -- there are six Joan files and
# coop_5 holds a real score (14000), so a glob or a [:5] slice would
# silently pick up the wrong set.


@pytest.mark.corpus
@pytest.mark.parametrize("name", _JOAN_SENTINEL_FILES)
def test_browsing_a_sentinel_score_saves_byte_identically(name, tmp_path) -> None:
    """The real instance of the sentinel-substitution case, not a synthetic
    one: required_score_for_score_victory is 0xFFFFFFFF on exactly these
    five 2_Joan_coop_* files.

    victory_score is a real QSpinBox showing 14000 (option_fields.py's
    sentinel_display) with the substitution tooltip set, not the as-stored
    QLabel this row used to build -- and browsing without touching it must
    still save byte-identically, the same shape as
    test_browsing_every_row_saves_byte_identically.
    """
    from PyQt5.QtWidgets import QSpinBox

    from tests.conftest import ROOT
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    path = ROOT / "examples" / name
    if not path.is_file():
        pytest.skip(f"corpus file not present: {path}")

    loaded = load_map_and_units(path)
    retriever = loaded._scenario.sections["GlobalVictory"].retriever_map[
        "required_score_for_score_victory"
    ]
    assert int(retriever.data) == 0xFFFFFFFF, f"{name} no longer carries the sentinel"

    baseline = tmp_path / "baseline.aoe2scenario"
    write_scenario(load_map_and_units(path), baseline)

    window = _options_window(path)
    try:
        panel = window.map_options_panel
        widget = panel.widget_for("victory_score")
        assert isinstance(widget, QSpinBox), f"built a {type(widget).__name__}, not a spinbox"
        assert widget.value() == 14000
        assert "Stored as unset" in widget.toolTip()
        assert panel.current_values()["victory_score"] == 0xFFFFFFFF

        widget.setFocus()
        widget.setValue(widget.value())

        assert window.option_edits is None, "browsing built an options model"
        browsed = tmp_path / "browsed.aoe2scenario"
        write_scenario(
            window.scenario, browsed, triggers=window.trigger_edits, options=window.option_edits
        )
        assert browsed.read_bytes() == baseline.read_bytes()
    finally:
        _close(window)


@pytest.mark.corpus
@pytest.mark.parametrize("name", _JOAN_SENTINEL_FILES)
def test_setting_a_sentinel_score_writes_the_real_value(name, tmp_path, monkeypatch) -> None:
    """The other half: typing a value into a sentinel row is a genuine edit,
    end to end through ViewerWindow.save_as() -- the reload-and-read-back
    shape test_an_option_edit_reads_back_after_a_reload uses in
    tests/test_options_write_path.py. Byte-locality for a GlobalVictory
    field is already covered by
    test_a_victory_condition_edit_changes_exactly_that_fields_bytes; this
    test only needs the round-tripped value.
    """
    from PyQt5.QtWidgets import QFileDialog

    from tests.conftest import ROOT
    from descape import option_fields
    from descape.scenario_io import load_map_and_units

    path = ROOT / "examples" / name
    if not path.is_file():
        pytest.skip(f"corpus file not present: {path}")

    out = tmp_path / "saved.aoe2scenario"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))

    window = _options_window(path)
    try:
        widget = _row(window, "victory_score")
        widget.setValue(5000)
        window.save_as()
        assert not window.edit_history.is_dirty, "save_as() did not mark the document saved"
    finally:
        _close(window)

    reloaded = load_map_and_units(out)
    spec = next(s for s in option_fields.specs_for(reloaded) if s.field_id == "victory_score")
    assert option_fields.current_value(reloaded, spec) == 5000


def test_every_widget_fits_the_pane_it_is_given() -> None:
    """The measured version of "does the form fit", not an eyeballed one.
    VictoryCondition and SecondaryGameMode are the same class of long-label
    enum whose 430 px combo forced TriggerPanel's own width fix."""
    from PyQt5.QtWidgets import QApplication

    from descape.map_options_panel import MapOptionsPanel

    window = _options_window(TRIGGER_FIXTURE)
    try:
        panel = window.map_options_panel
        QApplication.processEvents()
        for spec in panel._specs:
            widget = panel.widget_for(spec.field_id)
            assert widget.minimumSizeHint().width() <= MapOptionsPanel.MIN_USEFUL_WIDTH, (
                f"{spec.field_id}'s editor demands "
                f"{widget.minimumSizeHint().width()} px of a "
                f"{MapOptionsPanel.MIN_USEFUL_WIDTH} px panel"
            )
        assert not panel.area.horizontalScrollBar().isVisible()
    finally:
        _close(window)


# --- browsing is not editing ------------------------------------------------


def test_browsing_every_row_reports_no_edit() -> None:
    """Setting every row to the value it already shows must report nothing.

    This is what tabbing through the form does, and it is now load-bearing:
    the widgets are enabled, so a missing guard in _changed() would record one
    phantom undo step per field. It was explicitly vacuous while the panel
    shipped read-only -- a disabled widget emits no signal, so it passed with
    the guards deleted.
    """
    from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel

    reported = []
    window = _options_window(TRIGGER_FIXTURE)
    try:
        panel = window.map_options_panel
        panel._on_option_field = lambda spec, value: reported.append((spec.field_id, value))
        assert panel._editable_fields, "no row is editable -- this would pass vacuously"
        for spec in panel._specs:
            widget = panel.widget_for(spec.field_id)
            if isinstance(widget, QLabel):
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(widget.isChecked())
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(widget.currentIndex())
            else:
                widget.setValue(widget.value())
            widget.setFocus()
            QApplication.processEvents()
        assert reported == [], f"browsing reported edits: {reported}"
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_browsing_every_row_saves_byte_identically(tmp_path) -> None:
    """The acceptance gate this whole panel is measured by, and the analogue of
    tests/test_trigger_panel.py's own.

    A phantom record would not just add an undo step: it would dirty the
    document *and* flip a model's has_edits, reserialising a Triggers section
    on a file the user only looked at. Run against the trigger fixture rather
    than the blank template for exactly that reason.
    """
    from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel

    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    baseline = tmp_path / "baseline.aoe2scenario"
    write_scenario(load_map_and_units(TRIGGER_FIXTURE), baseline)

    window = _options_window(TRIGGER_FIXTURE)
    try:
        panel = window.map_options_panel
        for spec in panel._specs:
            widget = panel.widget_for(spec.field_id)
            if isinstance(widget, QLabel):
                continue
            widget.setFocus()
            if isinstance(widget, QCheckBox):
                widget.setChecked(widget.isChecked())
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(widget.currentIndex())
            else:
                widget.setValue(widget.value())
            QApplication.processEvents()

        assert window.option_edits is None, "browsing built an options model"
        assert window.trigger_edits is None, "browsing built a trigger model"
        assert not window.edit_history.is_dirty

        browsed = tmp_path / "browsed.aoe2scenario"
        write_scenario(
            window.scenario, browsed, triggers=window.trigger_edits, options=window.option_edits
        )
        assert browsed.read_bytes() == baseline.read_bytes()
    finally:
        _close(window)


def _editable_panel(loaded):
    """A standalone panel with every row editable, recording what it reports.

    Constructed directly rather than through a window: the panel is dumb by
    design, so driving it without one is what isolates _changed()'s own guards
    from the window's gates. Every field id is declared editable regardless of
    what the real gates would say about this file, which is the point -- the
    gates are tested separately, below.
    """
    from descape import option_fields
    from descape.map_options_panel import MapOptionsPanel

    conftest.ensure_qapp()
    reported = []
    panel = MapOptionsPanel(on_option_field=lambda spec, value: reported.append((spec.field_id, value)))
    panel.show_scenario(
        loaded,
        editable_fields=[spec.field_id for spec in option_fields.specs_for(loaded)],
    )
    return panel, reported


def test_an_editable_panel_reports_a_real_change_once() -> None:
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        widget = panel.widget_for("collide_and_correct")
        before = panel.current_values()["collide_and_correct"]
        widget.setChecked(not before)
        assert reported == [("collide_and_correct", int(not before))]
        # And the panel's own live value moved with it, so a second identical
        # set is a no-op rather than a second undo step.
        widget.setChecked(not before)
        assert reported == [("collide_and_correct", int(not before))]
    finally:
        panel.deleteLater()


def test_changed_ignores_a_value_equal_to_the_one_already_shown() -> None:
    """The equality guard, called directly rather than driven through a widget.

    Directly on purpose. Measured by mutation: deleting this guard leaves the
    whole file green when the panel is driven through its widgets, because Qt
    already de-duplicates an identical setChecked/setCurrentIndex/setValue,
    and this panel has no QLineEdit, so nothing here emits the focus-out
    editingFinished that makes the same guard fire constantly in TriggerPanel.
    The guard is still worth keeping -- an option record is pushed
    unconditionally, so anything that ever does reach _changed() with an
    unchanged value becomes a phantom undo step -- but a test claiming a
    widget path covers it would be measuring nothing.
    """
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        for spec in panel._specs:
            panel._changed(spec, panel.current_values()[spec.field_id])
        assert reported == [], f"a no-op change reported an edit: {reported}"
        assert len(panel._specs) >= 10, "the spec list shrank -- this walked almost nothing"
    finally:
        panel.deleteLater()


def test_changed_reports_nothing_while_populating() -> None:
    """The _populating guard, also called directly, and for the same reason:
    _build_widget() connects every signal *after* setting its value, so a
    populate emits nothing and the guard is unreachable through the widgets
    today (mutation-checked -- removing it leaves the file green).

    Kept because that connect-after-set ordering is the only thing making it
    unreachable, and because a populate now happens on every undo of an option
    edit -- against a form whose rows are enabled, so the ordering is the only
    thing left between a repopulate and a reported edit.
    """
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        spec = panel._specs[0]
        target = panel.current_values()[spec.field_id] + 1
        panel._populating = True
        panel._changed(spec, target)
        assert reported == []
        # The guard is the only thing stopping it: the same call with the flag
        # down does report, so this cannot be passing for an unrelated reason
        # (a read-only panel, an unknown field id, a value that was equal).
        panel._populating = False
        panel._changed(spec, target)
        assert reported == [(spec.field_id, target)]
    finally:
        panel.deleteLater()


def test_a_reported_edit_becomes_the_panels_live_value() -> None:
    """_changed() records what it reported, so a second edit compares against
    the value on screen rather than against the file's. Without it every edit
    after the first would be measured against a value the user has already
    replaced."""
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        spec = next(s for s in panel._specs if s.field_id == "ai_map_type")
        original = panel.current_values()["ai_map_type"]
        panel._changed(spec, original + 1)
        assert panel.current_values()["ai_map_type"] == original + 1
        # Reporting the same new value again is now a no-op...
        panel._changed(spec, original + 1)
        # ...and reverting to the file's original is a real edit, not a no-op.
        panel._changed(spec, original)
        assert reported == [
            ("ai_map_type", original + 1),
            ("ai_map_type", original),
        ]
    finally:
        panel.deleteLater()


def test_a_repopulate_reports_no_edit() -> None:
    """The behaviour both guards above exist for, through the real path:
    re-showing the same document must not look like the user changed
    anything."""
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        panel.show_scenario(loaded)
        assert reported == []
    finally:
        panel.deleteLater()


def test_pending_values_override_the_files_own(tmp_path) -> None:
    """show_scenario(values=...) is how the window repopulates a panel whose
    edits live in a model: neither write path mutates a retriever, so
    re-reading the file would show every pending edit reverted."""
    from descape.scenario_io import load_map_and_units
    from descape.map_options_panel import MapOptionsPanel

    conftest.ensure_qapp()
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    panel = MapOptionsPanel()
    try:
        panel.show_scenario(loaded)
        original = panel.current_values()["ai_map_type"]
        panel.show_scenario(loaded, values={"ai_map_type": original + 1})
        assert panel.current_values()["ai_map_type"] == original + 1
        assert panel.widget_for("ai_map_type").value() == original + 1
        # An unknown field_id is ignored rather than inventing a row.
        panel.show_scenario(loaded, values={"not_a_field": 7})
        assert "not_a_field" not in panel.current_values()
    finally:
        panel.deleteLater()


# --- editing, through the window --------------------------------------------


def _row(window, field_id):
    return window.map_options_panel.widget_for(field_id)


def test_editing_a_scalar_row_records_one_undo_step_and_dirties_the_document() -> None:
    window = _options_window(TRIGGER_FIXTURE)
    try:
        widget = _row(window, "collide_and_correct")
        before = window.map_options_panel.current_values()["collide_and_correct"]
        widget.setChecked(not before)

        assert window.option_edits is not None, "the edit built no options model"
        assert window.option_edits.current_value("collide_and_correct") == int(not before)
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[0].kind == "options"
        assert window.edit_history.is_dirty
        # And the trigger model stayed out of it -- a scalar row must not
        # reserialise the Triggers section.
        assert window.trigger_edits is None
    finally:
        _close(window)


def test_editing_the_exec_order_row_records_a_trigger_step_not_an_option_one() -> None:
    """The one row whose uniform look hides a different wiring. Routing it to
    the options model instead would leave trigger_edits None on a save whose
    only change was this flag, and write_scenario() would emit the original
    bytes with no error."""
    window = _options_window(TRIGGER_FIXTURE)
    try:
        widget = _row(window, "legacy_exec_order")
        before = window.map_options_panel.current_values()["legacy_exec_order"]
        widget.setCurrentIndex(widget.findData(1 - before))

        assert window.trigger_edits is not None, "the edit built no trigger model"
        assert window.trigger_edits.exec_order == 1 - before
        assert window.trigger_edits.has_edits, "the trigger model reads as clean"
        assert window.option_edits is None, "exec-order went through the options model"
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[0].kind == "trigger"
    finally:
        _close(window)


def test_undoing_a_scalar_edit_puts_the_row_back() -> None:
    """The repopulate matters as much as the model restore: neither write path
    mutates a retriever, so a repopulate that did not carry pending values
    would reset the form while the model still held the edit."""
    window = _options_window(TRIGGER_FIXTURE)
    try:
        before = window.map_options_panel.current_values()["collide_and_correct"]
        _row(window, "collide_and_correct").setChecked(not before)
        window.undo()

        assert window.map_options_panel.current_values()["collide_and_correct"] == before
        assert _row(window, "collide_and_correct").isChecked() == bool(before)
        assert not window.option_edits.has_edits
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_undoing_an_exec_order_edit_puts_the_row_back() -> None:
    """Undoing it produces a *trigger* record while the user is looking at the
    Map Options form, which is why _move_history repopulates on both kinds."""
    window = _options_window(TRIGGER_FIXTURE)
    try:
        widget = _row(window, "legacy_exec_order")
        before = window.map_options_panel.current_values()["legacy_exec_order"]
        widget.setCurrentIndex(widget.findData(1 - before))
        window.undo()

        assert window.map_options_panel.current_values()["legacy_exec_order"] == before
        assert _row(window, "legacy_exec_order").currentData() == before
        assert not window.trigger_edits.has_edits
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_an_undone_and_redone_edit_ends_where_it_started() -> None:
    """Also the discriminator for the pending-value pass in
    _show_map_options(): a redo repopulates the form while the model holds a
    value the retriever does not, so dropping `values=` shows the file's own
    number here and nowhere else. The undo tests above cannot catch it -- after
    an undo there is nothing pending for the two paths to disagree about.
    """
    window = _options_window(TRIGGER_FIXTURE)
    try:
        before = window.map_options_panel.current_values()["collide_and_correct"]
        _row(window, "collide_and_correct").setChecked(not before)
        window.undo()
        window.redo()
        assert window.map_options_panel.current_values()["collide_and_correct"] == int(not before)
        assert _row(window, "collide_and_correct").isChecked() == (not before)
        assert window.option_edits.has_edits
    finally:
        _close(window)


def test_a_repopulate_over_a_pending_edit_reports_nothing_further() -> None:
    """The phantom-record failure the pending-value pass exists to prevent.

    A repopulate that read the retrievers instead would show the file's own
    value while the model still held the edit, and _changed()'s equality guard
    compares against what the panel last showed -- so setting the row back to
    the value the model already has would report a *second* edit for a change
    the document had already made.
    """
    window = _options_window(TRIGGER_FIXTURE)
    try:
        before = window.map_options_panel.current_values()["collide_and_correct"]
        _row(window, "collide_and_correct").setChecked(not before)
        assert len(window.edit_history.records) == 1

        window._show_map_options()
        assert window.map_options_panel.current_values()["collide_and_correct"] == int(not before)
        assert _row(window, "collide_and_correct").isChecked() is not bool(before)
        _row(window, "collide_and_correct").setChecked(not before)
        assert len(window.edit_history.records) == 1, "the repopulate produced a phantom record"
    finally:
        _close(window)


def test_editing_then_saving_writes_the_edited_value(tmp_path, monkeypatch) -> None:
    """End to end through ViewerWindow.save_as(), not through a direct
    write_scenario() call.

    The distinction is the whole point: save_as() is the only place a real user
    reaches the write path, and its `options=` argument is what carries the
    edit. A test that calls write_scenario() itself passes with that argument
    deleted, and every map-option edit is then silently dropped on save --
    the same failure shape as wiring exec-order to the wrong model, one layer
    up. Mutation-checked by deleting the kwarg.
    """
    from PyQt5.QtWidgets import QFileDialog

    from descape import option_fields
    from descape.scenario_io import load_map_and_units

    out = tmp_path / "saved.aoe2scenario"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))

    window = _options_window(TRIGGER_FIXTURE)
    try:
        before = window.map_options_panel.current_values()["collide_and_correct"]
        _row(window, "collide_and_correct").setChecked(not before)
        window.save_as()
        assert not window.edit_history.is_dirty, "save_as() did not mark the document saved"
    finally:
        _close(window)

    reloaded = load_map_and_units(out)
    spec = next(s for s in option_fields.specs_for(reloaded) if s.field_id == "collide_and_correct")
    assert option_fields.current_value(reloaded, spec) == int(not before)


def test_saving_after_an_exec_order_edit_writes_the_flag(tmp_path, monkeypatch) -> None:
    """The other half of the same argument: exec-order rides on `triggers=`,
    which save_as() has passed since 4b, but nothing had yet driven a
    Map-Options-originated trigger edit through it."""
    from PyQt5.QtWidgets import QFileDialog

    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.trigger_model import TriggerEditModel

    out = tmp_path / "saved.aoe2scenario"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))

    window = _options_window(TRIGGER_FIXTURE)
    try:
        widget = _row(window, "legacy_exec_order")
        before = window.map_options_panel.current_values()["legacy_exec_order"]
        widget.setCurrentIndex(widget.findData(1 - before))
        window.save_as()
    finally:
        _close(window)

    reloaded = load_map_and_units(out)
    parse_triggers(reloaded)
    assert TriggerEditModel(reloaded).exec_order == 1 - before


def test_an_as_stored_row_is_never_editable() -> None:
    """A value no editor could show truthfully is a QLabel, which emits no
    signal -- so no write path can reach it. That is the containment, and it is
    what keeps a save from writing 1 over villager_force_drop's stored 90."""
    from PyQt5.QtWidgets import QLabel

    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    parsed = loaded._scenario.sections["Map"].retriever_map["villager_force_drop"]
    original = parsed.data
    parsed.data = 90
    panel, reported = _editable_panel(loaded)
    try:
        widget = panel.widget_for("villager_force_drop")
        assert isinstance(widget, QLabel), "an unrepresentable value built an editor"
        assert reported == []
    finally:
        parsed.data = original
        panel.deleteLater()


# --- geometry ---------------------------------------------------------------


def test_the_status_line_stays_short_enough_to_leave_the_form_visible() -> None:
    """The regression an offscreen capture found and every width/height
    assertion missed.

    Spelling all four explanations out inline grew this word-wrapped label to
    209 px of a 760 px panel and pushed the Triggers group below the fold --
    the single row one of those explanations was about, hidden by the note
    about it. Nothing was overlapping or squeezed, so the geometry checks were
    all green. Counts belong on the label; the prose belongs in its tooltip.
    """
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.map_options_panel import MapOptionsPanel
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    parse_triggers(loaded)
    parsed = loaded._scenario.sections["Map"].retriever_map["villager_force_drop"]
    original = parsed.data
    parsed.data = 90  # forces the as-stored note on as well

    panel = MapOptionsPanel()
    try:
        panel.resize(MapOptionsPanel.MIN_USEFUL_WIDTH, 760)
        # Every note at once: the read-only count, the absent note, the
        # as-stored note and a gate note. This is the worst case, not a typical
        # one.
        panel.show_scenario(
            loaded,
            editable_fields=["collide_and_correct"],
            read_only_reasons={"legacy_exec_order": ViewerWindow._EXEC_ORDER_READ_ONLY},
            notes=["Trigger execution order is read-only for this file: its Triggers "
                   "section failed the alignment gate that a save would splice through."],
        )
        panel.show()
        QApplication.processEvents()

        assert panel.status.height() <= panel.height() // 6, (
            f"the status label takes {panel.status.height()} px of {panel.height()}"
        )
        # And the detail is not lost, only moved.
        assert "read-only" in panel.status.text()
        assert MapOptionsPanel._ABSENT_NOTE in panel.status.toolTip()
        assert "execution order" in panel.status.toolTip()
    finally:
        parsed.data = original
        panel.deleteLater()


def test_every_group_renders_at_the_height_its_layout_asks_for() -> None:
    """The half of the plan's in-session visual check this panel was missing.

    tests/test_trigger_panel.py has the same check because a wrapped row's
    height depends on its width while a QScrollArea sizes its widget from a
    sizeHint computed as if nothing wrapped -- every row was squeezed to 6 px
    and drew over the next one while host width, scrollbar state and field
    widths all still measured correct. activate() is load-bearing: without it
    the measurement answers for the previous form.
    """
    from PyQt5.QtWidgets import QApplication, QGroupBox

    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.map_options_panel import MapOptionsPanel

    conftest.ensure_qapp()
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    parse_triggers(loaded)

    panel = MapOptionsPanel()
    try:
        panel.resize(MapOptionsPanel.MIN_USEFUL_WIDTH, 760)
        panel.show_scenario(loaded, editable_fields=[s.field_id for s in
                                                     __import__("descape.option_fields",
                                                                fromlist=["x"]).specs_for(loaded)])
        panel.show()
        QApplication.processEvents()
        panel.host_layout.activate()
        QApplication.processEvents()

        boxes = panel.host.findChildren(QGroupBox)
        assert boxes, "no group boxes built -- this would pass vacuously"
        for box in boxes:
            wanted = box.layout().heightForWidth(box.width())
            if wanted > 0:
                assert box.height() >= wanted, (
                    f"{box.title()} renders {box.height()} px but its layout wants {wanted}"
                )
        for field_id, widget in panel._widgets.items():
            assert widget.height() >= widget.minimumSizeHint().height(), (
                f"{field_id} is {widget.height()} px, needs "
                f"{widget.minimumSizeHint().height()}"
            )
    finally:
        panel.deleteLater()


# --- the two gates ----------------------------------------------------------


def test_the_two_editable_gates_are_resolved_independently(monkeypatch) -> None:
    """options_model says nothing about exec-order and trigger_write_supported
    says nothing about the scalars, so a file can legitimately land on either
    side of one and the other side of the other."""
    window = _options_window(TRIGGER_FIXTURE)
    try:
        editable = window._editable_option_fields()
        assert "legacy_exec_order" in editable
        assert "lock_teams" in editable

        monkeypatch.setattr(window.scenario, "trigger_write_supported", False)
        only_scalars = window._editable_option_fields()
        assert "legacy_exec_order" not in only_scalars
        assert "lock_teams" in only_scalars, "the trigger gate took the scalars with it"
    finally:
        _close(window)


def test_an_unverifiable_options_block_greys_the_scalars_but_not_exec_order(
    monkeypatch,
) -> None:
    window = _options_window(TRIGGER_FIXTURE)
    try:
        monkeypatch.setattr(window.scenario, "terrain_block_offset", -1)
        editable = window._editable_option_fields()
        assert editable == frozenset({"legacy_exec_order"}), editable
    finally:
        _close(window)


def test_a_greyed_exec_order_row_says_why(monkeypatch) -> None:
    """A greyed row must carry its own tooltip, not only a status-line note.

    The exec-order row is the case that forced this: it has no spec tooltip of
    its own, so a user hovering it learned nothing at all. Plan item 17 named
    the tooltip specifically, on the Show sprites precedent.
    """
    from descape.viewer import ViewerWindow

    window = _options_window(TRIGGER_FIXTURE)
    try:
        assert window._map_options_notes() == []
        assert window._map_options_read_only_reasons() == {}
        assert not _row(window, "legacy_exec_order").toolTip()

        monkeypatch.setattr(window.scenario, "trigger_write_supported", False)
        window._show_map_options()

        row = _row(window, "legacy_exec_order")
        assert not row.isEnabled()
        assert row.toolTip() == ViewerWindow._EXEC_ORDER_READ_ONLY
        # In the status *tooltip*, not its label -- see _status_text().
        assert "execution order" in window.map_options_panel.status.toolTip()
        # The scalars keep their own gate's wording, not this one's.
        assert not _row(window, "collide_and_correct").toolTip()
    finally:
        _close(window)


def test_a_greyed_scalar_row_says_why_in_its_own_words(monkeypatch) -> None:
    """The other gate's text. Two reasons rather than one for the same reason
    the gates are separate: a row greyed because the Map anchor failed is a
    different situation from one greyed because the Triggers section did."""
    from descape.viewer import ViewerWindow

    window = _options_window(TRIGGER_FIXTURE)
    try:
        monkeypatch.setattr(window.scenario, "terrain_block_offset", -1)
        window._show_map_options()
        assert _row(window, "collide_and_correct").toolTip() == ViewerWindow._SCALARS_READ_ONLY
        # exec-order is still editable on this file, so it gets no reason.
        assert _row(window, "legacy_exec_order").isEnabled()
        assert not _row(window, "legacy_exec_order").toolTip()
    finally:
        _close(window)


def test_a_read_only_reason_never_displaces_an_as_stored_explanation() -> None:
    """An as-stored label's tooltip accounts for the raw number the file holds,
    which is the more specific explanation of the two and the one a user cannot
    reconstruct from anywhere else."""
    from PyQt5.QtWidgets import QLabel

    from descape.scenario_io import load_map_and_units
    from descape.map_options_panel import MapOptionsPanel

    conftest.ensure_qapp()
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    parsed = loaded._scenario.sections["Map"].retriever_map["villager_force_drop"]
    original = parsed.data
    parsed.data = 90
    panel = MapOptionsPanel()
    try:
        panel.show_scenario(
            loaded,
            editable_fields=(),
            read_only_reasons={"villager_force_drop": "a gate refused this row"},
        )
        widget = panel.widget_for("villager_force_drop")
        assert isinstance(widget, QLabel)
        assert "90" in widget.toolTip() and "a gate refused this row" not in widget.toolTip()
    finally:
        parsed.data = original
        panel.deleteLater()


def test_a_fully_read_only_file_reports_it_once_rather_than_per_row(monkeypatch) -> None:
    from descape.scenario_io import load_map_and_units
    from descape.map_options_panel import MapOptionsPanel

    conftest.ensure_qapp()
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    panel = MapOptionsPanel()
    try:
        panel.show_scenario(loaded, editable_fields=())
        assert "all read-only" in panel.status.text()
        assert MapOptionsPanel._ALL_READ_ONLY_NOTE in panel.status.toolTip()
        for spec in panel._specs:
            assert _is_read_only(panel.widget_for(spec.field_id))
    finally:
        panel.deleteLater()


# --- corpus -----------------------------------------------------------------


@pytest.mark.corpus
def test_map_options_panel_populates_every_corpus_file(scenario_path) -> None:
    """The tier that covers the absent-field branches at all.

    Both default-tier fixtures are 1.58/trigger-4.9, so every row is present
    on them -- villager_force_drop (>= 1.37), lock_coop_alliances and
    secondary_game_modes (>= 1.42), ai_map_type (absent on every 1.41 file)
    and legacy_exec_order (needs trigger version >= 4.5) only ever go missing
    out here. A file is allowed to be short of rows; it is not allowed to
    render a row whose widget disagrees with the value the file holds.
    """
    from PyQt5.QtWidgets import QLabel

    from descape import option_fields
    from descape.map_options_panel import MapOptionsPanel

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Map Options")
        panel = window.map_options_panel
        assert panel._loaded is window.scenario

        # Map's collide_and_correct / no_waves_on_shore, Options' all_techs
        # and Global Victory's three live fields (mode,
        # required_score_for_score_victory, time_for_timed_game_...) exist in
        # every version the corpus carries, so a file with fewer rows than
        # that has lost something rather than merely predating a field. The
        # rest are version-gated: villager_force_drop (>= 1.37),
        # lock_coop_alliances / secondary_game_modes (>= 1.42), ai_map_type
        # (absent on every 1.41 file) and legacy_exec_order (needs trigger
        # version >= 4.5). Diplomacy's four scalars moved to DiplomacyPanel
        # (step 5) and are no longer among panel._specs here at all -- see
        # tests/test_diplomacy_panel.py's own corpus coverage for them.
        # 7 is the measured minimum across examples/ (the 20-file corpus
        # this environment carries), not a guess: the six C2_ElCid_coop_*
        # files and 0_June_Event_Scenario lose all four version-gated rows
        # at once (ai_map_type, lock_coop_alliances, secondary_game_modes,
        # legacy_exec_order), landing at exactly 11 - 4 = 7.
        assert len(panel._specs) >= 7, (
            f"{scenario_path.name} built only {len(panel._specs)} rows"
        )
        for spec in panel._specs:
            widget = panel.widget_for(spec.field_id)
            raw = int(option_fields.current_value(window.scenario, spec))
            shown = _shown_value(panel, spec)
            expected = (
                spec.sentinel_display
                if spec.sentinel is not None and raw == spec.sentinel
                else raw
            )
            assert shown == expected, (
                f"{scenario_path.name}: {spec.field_id} shows {shown}, expected {expected} "
                f"(file holds {raw})"
            )
            assert widget.minimumSizeHint().width() <= MapOptionsPanel.MIN_USEFUL_WIDTH, (
                f"{scenario_path.name}: {spec.field_id}'s editor demands "
                f"{widget.minimumSizeHint().width()} px"
            )
            # Three states, not two: an as-stored row is a QLabel and is
            # read-only whatever the gate says, which is the containment
            # itself. villager_force_drop on the 1.41 files is the live case --
            # the byte verifies and packs, so the gate lists it as editable,
            # and the label is what stops a save writing 1 over its stored 90.
            if isinstance(widget, QLabel):
                continue
            assert widget.isEnabled() is (spec.field_id in panel._editable_fields), (
                f"{scenario_path.name}: {spec.field_id} disagrees with its gate"
            )

        # The exec-order row's presence follows the trigger parse, not the
        # scenario version -- finding 2 of the plan: F7_3_York is 1.55 and
        # still has no exec-order byte.
        from descape.scenario_io import parse_triggers

        section = window.scenario._scenario.sections.get("Triggers")
        expected = (
            parse_triggers(window.scenario) is not None
            and section is not None
            and "legacy_exec_order" in section.retriever_map
            and scenario_io_retriever_length(section.retriever_map["legacy_exec_order"]) == 1
        )
        assert ("legacy_exec_order" in panel.current_values()) is expected
    finally:
        _close(window)


def scenario_io_retriever_length(retriever) -> int:
    from descape.scenario_io import retriever_length

    return retriever_length(retriever)


@pytest.mark.corpus
def test_browsing_map_options_leaves_the_document_clean(scenario_path, tmp_path) -> None:
    """The corpus half of the acceptance gate: entering the mode and walking
    every widget on a real file must neither dirty the document nor build
    either edit model, and the resulting save must be byte-identical.

    The Triggers parse the mode performs is what makes this worth running out
    here rather than only on the fixtures: a model flipped into existence by
    the parse would reserialise the Triggers section on a file the user only
    looked at, and the corpus is where the big ones are.
    """
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Map Options")
        for spec in window.map_options_panel._specs:
            window.map_options_panel.widget_for(spec.field_id).setFocus()
            QApplication.processEvents()
        assert not window.edit_history.is_dirty
        assert window.trigger_edits is None, "browsing built a trigger edit model"
        assert window.option_edits is None, "browsing built an options edit model"

        if not window.scenario.terrain_write_supported:
            return  # no save path at all for this file, by design
        baseline = tmp_path / "baseline.aoe2scenario"
        write_scenario(load_map_and_units(scenario_path), baseline)
        browsed = tmp_path / "browsed.aoe2scenario"
        write_scenario(
            window.scenario, browsed, triggers=window.trigger_edits, options=window.option_edits
        )
        assert browsed.read_bytes() == baseline.read_bytes(), scenario_path.name
    finally:
        _close(window)
