"""Coverage for Players mode and PlayersPanel.

Same offscreen-ViewerWindow technique as tests/test_trigger_panel.py and
tests/test_map_options_panel.py. Editable as of step 3c, for whichever
fields both write-path gates (options_write_supported()/
players_write_supported()) allow -- see PlayersPanel's own module
docstring. "Browsing never dirties the document" stays an acceptance gate
(the widgets a gate refuses fire no signal at all), alongside "an editable
row's edit round-trips through the same OptionsEditModel/OptionsDiffRecord
Map Options and Diplomacy use" -- see tests/test_player_options_write_path.py
for the byte-level half of that claim; this file covers the panel/window
wiring on top of it.
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


def _window():
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.resize(1500, 900)
    window.show()
    QApplication.processEvents()
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _shown_value(panel, spec):
    from PyQt5.QtWidgets import QCheckBox, QComboBox, QLabel, QLineEdit, QSpinBox

    widget = panel.widget_for(spec.field_id)
    if isinstance(widget, QCheckBox):
        return int(widget.isChecked())
    if isinstance(widget, QComboBox):
        return widget.currentData()
    if isinstance(widget, QLineEdit):
        # tribe_name, editable -- the one TEXT spec that isn't a plain label.
        return widget.text()
    if isinstance(widget, QLabel):
        # Either the TEXT branch or the out-of-range branch -- both render
        # as a plain label, read verbatim off its text.
        return widget.text()
    assert isinstance(widget, QSpinBox), f"{spec.field_id} built a {type(widget).__name__}"
    return widget.value()


def _expected_display(spec, raw):
    """What `_shown_value` should equal for `raw`. civilization/architecture
    are COMBO-kind as of Step A, so -- same as any other COMBO field
    (starting_age, color, player_type) -- the widget's currentData() is the
    stored value verbatim; no resolved-name translation happens here."""
    return raw


def _players_window(path=BLANK_FIXTURE):
    window = _window()
    window.load_scenario(path)
    window.mode_combo.setCurrentText("Players")
    return window


# --- the mode ----------------------------------------------------------


def test_players_mode_swaps_the_left_panel_and_gates_edit_tools() -> None:
    from descape.viewer import _LEFT_PAGE_PLAYERS

    window = _window()
    try:
        assert window.left_stack.currentIndex() == 0
        window.mode_combo.setCurrentText("Players")
        assert window.mode == "players"
        assert window.left_stack.currentIndex() == _LEFT_PAGE_PLAYERS == 4
        assert not window.draw_action.isEnabled()
        assert not window.elevation_action.isEnabled()
        assert window.pan_action.isChecked()

        window.mode_combo.setCurrentText("View")
        assert window.left_stack.currentIndex() == 0
    finally:
        _close(window)


def test_players_mode_appended_to_the_combo() -> None:
    window = _window()
    try:
        modes = [window.mode_combo.itemText(i) for i in range(window.mode_combo.count())]
        assert modes == [
            "View",
            "Terrain",
            "Units",
            "Triggers",
            "Map Options",
            "Players",
            "Diplomacy",
            "Messages",
        ]
    finally:
        _close(window)


def test_no_document_shows_the_empty_state() -> None:
    window = _window()
    try:
        window.mode_combo.setCurrentText("Players")
        panel = window.players_panel
        assert panel.status.text() == panel._NO_DOCUMENT
        assert not panel.player_combo.isEnabled()
    finally:
        _close(window)


def test_closing_the_document_clears_the_panel() -> None:
    window = _players_window()
    try:
        panel = window.players_panel
        assert panel.player_combo.isEnabled()
        window.close_scenario()
        assert panel.status.text() == panel._NO_DOCUMENT
        assert not panel.player_combo.isEnabled()
        assert panel.player_combo.count() == 0
    finally:
        _close(window)


# --- the form ------------------------------------------------------------


def test_player_selector_lists_p1_through_p8_with_swatches() -> None:
    window = _players_window()
    try:
        panel = window.players_panel
        assert panel.player_combo.count() == 8
        labels = [panel.player_combo.itemText(i) for i in range(8)]
        assert labels == [f"P{n}" for n in range(1, 9)]
        for i in range(8):
            assert not panel.player_combo.itemIcon(i).isNull()
        assert panel.player_combo.currentIndex() == 0
    finally:
        _close(window)


def test_select_player_happy_path() -> None:
    window = _players_window()
    try:
        panel = window.players_panel
        assert panel.select_player(6) is True
        assert panel.player_combo.currentIndex() == 5

        # Already-current player is a no-op that still reports success.
        assert panel.select_player(6) is True
        assert panel.player_combo.currentIndex() == 5
    finally:
        _close(window)


def test_select_player_rejects_gaia() -> None:
    window = _players_window()
    try:
        panel = window.players_panel
        index_before = panel.player_combo.currentIndex()
        assert panel.select_player(0) is False
        assert panel.player_combo.currentIndex() == index_before
    finally:
        _close(window)


def test_select_player_rejects_out_of_range_with_no_document() -> None:
    window = _window()
    try:
        panel = window.players_panel
        assert not panel.player_combo.isEnabled()
        assert panel.select_player(3) is False
    finally:
        _close(window)


def test_groups_match_the_plan() -> None:
    window = _players_window()
    try:
        panel = window.players_panel
        groups = []
        for spec in panel._specs:
            if spec.group not in groups:
                groups.append(spec.group)
        # Order comes from player_fields._SPECS -- Identity, Start, AI, then
        # Point of View (the GAIA_FIRST exercise the maintainer plan calls
        # for including).
        assert groups == ["Identity", "Start", "AI", "Point of View"]
    finally:
        _close(window)


def test_tier1_rows_are_enabled_when_both_gates_hold() -> None:
    """The blank fixture passes both options_write_supported() and
    players_write_supported(), so every Tier-1 field -- including
    tribe_name, as of step 3d -- should be enabled. Tier 2 and player_type
    stay disabled regardless, as facts about the field, not the file."""
    from PyQt5.QtWidgets import QLabel

    window = _players_window()
    try:
        panel = window.players_panel
        editable = window._editable_player_fields()
        assert editable, "no field is editable -- this would pass vacuously"
        assert "tribe_name" in editable
        assert {"civilization", "architecture"} <= editable  # Step B: writable on every version
        assert {"player_type", "personality"}.isdisjoint(editable)
        for spec in panel._specs:
            widget = panel.widget_for(spec.field_id)
            if isinstance(widget, QLabel):
                assert spec.field_id not in editable, f"{spec.field_id} is a label but editable"
                continue
            assert widget.isEnabled() == (spec.field_id in editable), spec.field_id
    finally:
        _close(window)


def test_civilization_is_a_combo_that_selects_the_stored_str_value() -> None:
    """BLANK_FIXTURE is 1.58 (str16), so the stored value is a Civilization
    string like 'RANDOM-CIV' -- confirm findData()/setCurrentIndex() with a
    Python str actually selects that row rather than silently falling back
    to index 0 (the first civ alphabetically), which would look identical
    to a passing test unless currentData() is checked against the raw
    value, not just widget type. Step B: also confirm this row is
    editable, unlike the read-only "unwritable in Step A" state this test
    used to pin."""
    from PyQt5.QtWidgets import QComboBox

    from descape.player_fields import current_value

    window = _players_window()
    try:
        panel = window.players_panel
        spec = next(s for s in panel._specs if s.field_id == "civilization")
        widget = panel.widget_for("civilization")
        assert isinstance(widget, QComboBox)
        raw = current_value(window.scenario, spec, 1)
        assert isinstance(raw, str)
        assert widget.currentData() == raw
        assert widget.isEnabled()
    finally:
        _close(window)


def test_civilization_choices_exclude_gaia() -> None:
    from PyQt5.QtWidgets import QComboBox

    window = _players_window()
    try:
        panel = window.players_panel
        widget = panel.widget_for("civilization")
        assert isinstance(widget, QComboBox)
        all_data = {widget.itemData(i) for i in range(widget.count())}
        assert "GAIA" not in all_data
    finally:
        _close(window)


def test_player_type_carries_its_own_reason() -> None:
    window = _players_window()
    try:
        panel = window.players_panel
        assert "unconfirmed" in panel.widget_for("player_type").toolTip().lower()
    finally:
        _close(window)


def test_tribe_name_is_a_line_edit_once_both_gates_hold() -> None:
    """tribe_name is the one TEXT spec that becomes a QLineEdit rather than
    a read-only QLabel once the write path verifies -- see
    PlayersPanel._build_widget's TEXT branch."""
    from PyQt5.QtWidgets import QLineEdit

    window = _players_window()
    try:
        panel = window.players_panel
        widget = panel.widget_for("tribe_name")
        assert isinstance(widget, QLineEdit)
        assert widget.isEnabled()
    finally:
        _close(window)


def test_switching_player_repopulates_the_form_with_that_players_values() -> None:
    from descape.player_fields import current_value

    window = _players_window()
    try:
        panel = window.players_panel
        for player_id in (1, 8):
            panel.player_combo.setCurrentIndex(player_id - 1)
            assert panel._player_id == player_id
            for spec in panel._specs:
                raw = current_value(window.scenario, spec, player_id)
                shown = _shown_value(panel, spec)
                assert str(shown) == str(_expected_display(spec, raw)), (
                    f"P{player_id} {spec.field_id}: shows {shown!r}, file holds {raw!r}"
                )
    finally:
        _close(window)


def test_panel_source_never_references_ai_files() -> None:
    """ai_files holds the entire embedded .ai script source (up to several MB
    across the corpus) -- see player_fields.py and the maintainer plan. A
    single row referencing it could dump megabytes into the form."""
    import descape.players_panel as players_panel_module

    source = Path(players_panel_module.__file__).read_text()
    assert "ai_files" not in source


def test_every_widget_fits_min_useful_width() -> None:
    from descape.players_panel import PlayersPanel

    window = _players_window()
    try:
        panel = window.players_panel
        for spec in panel._specs:
            widget = panel.widget_for(spec.field_id)
            width = widget.minimumSizeHint().width()
            assert width <= PlayersPanel.MIN_USEFUL_WIDTH, f"{spec.field_id} demands {width} px"
    finally:
        _close(window)


# --- read-only containment -----------------------------------------------


def test_browsing_every_row_leaves_the_document_clean() -> None:
    """Walking every row on every player must never dirty the document --
    there is no callback for a disabled widget to fire in the first place,
    but this is the acceptance gate that keeps it that way."""
    from PyQt5.QtWidgets import QApplication

    window = _players_window()
    try:
        panel = window.players_panel
        for player_index in range(8):
            panel.player_combo.setCurrentIndex(player_index)
            for spec in panel._specs:
                widget = panel.widget_for(spec.field_id)
                widget.setFocus()
                QApplication.processEvents()
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_opening_a_second_map_while_in_the_mode_repopulates() -> None:
    """Its own if/else in load_scenario, not chained onto Map Options' --
    see the plan's touch list. Opening a second file while already in
    Players mode must show the new file's data, not the previous one's."""
    from AoE2ScenarioParser.datasets.object_support import StartingAge

    from descape.player_fields import current_value

    window = _players_window()
    try:
        panel = window.players_panel
        spec = next(s for s in panel._specs if s.field_id == "starting_age")
        first_value = current_value(window.scenario, spec, 1)

        window.load_scenario(BLANK_FIXTURE)
        assert panel._loaded is window.scenario
        second_value = current_value(window.scenario, spec, 1)
        assert second_value == first_value  # same fixture -- sanity check
        shown = _shown_value(panel, spec)
        assert str(shown) == str(second_value)
        assert isinstance(second_value, int) and second_value in {m.value for m in StartingAge}
    finally:
        _close(window)


# --- editing (step 3c) ----------------------------------------------------


def _editable_panel(loaded):
    """A standalone panel with every Tier-1 field editable, recording what
    it reports -- same technique test_map_options_panel.py's own
    _editable_panel() uses, isolating _changed()'s own guards from the
    window's gates."""
    from descape.player_fields import specs_for
    from descape.players_panel import PlayersPanel

    conftest.ensure_qapp()
    reported = []
    panel = PlayersPanel(
        on_player_field=lambda spec, player_id, value: reported.append((spec.field_id, player_id, value))
    )
    panel.show_scenario(loaded, editable_fields=[s.field_id for s in specs_for(loaded)])
    return panel, reported


def test_an_editable_panel_reports_a_real_change_once() -> None:
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(BLANK_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        widget = panel.widget_for("base_priority")
        before = panel.current_values()["base_priority"]
        widget.setValue(before + 1)
        assert reported == [("base_priority", 1, before + 1)]
        widget.setValue(before + 1)  # same value again -- a no-op
        assert reported == [("base_priority", 1, before + 1)]
    finally:
        panel.deleteLater()


def test_tribe_name_commits_on_editing_finished_not_on_every_keystroke() -> None:
    """QLineEdit.editingFinished, not textChanged -- one undo record per
    edit rather than one per keystroke (step 3d)."""
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(BLANK_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        widget = panel.widget_for("tribe_name")
        widget.setText("Franks")  # textChanged alone must report nothing
        assert reported == []
        widget.editingFinished.emit()
        assert reported == [("tribe_name", 1, "Franks")]
    finally:
        panel.deleteLater()


def test_tribe_name_validator_rejects_text_past_the_encoded_byte_budget() -> None:
    """The slot is 256 bytes with one reserved for the NUL terminator --
    255 encoded bytes is the real ceiling, not 255 characters, since
    MAIN_CHARSET (utf-8) is multi-byte for non-ASCII input."""
    from PyQt5.QtGui import QValidator

    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(BLANK_FIXTURE)
    panel, _reported = _editable_panel(loaded)
    try:
        widget = panel.widget_for("tribe_name")
        validator = widget.validator()
        assert validator.validate("x" * 255, 255)[0] == QValidator.Acceptable
        assert validator.validate("x" * 256, 256)[0] == QValidator.Invalid
        # "é" is two bytes in utf-8, so 128 of them is 256 encoded bytes --
        # already past the 255-byte budget despite being half as many
        # characters as the all-ASCII case above.
        assert validator.validate("é" * 128, 128)[0] == QValidator.Invalid
    finally:
        panel.deleteLater()


def test_changed_ignores_a_value_equal_to_the_one_already_shown() -> None:
    """The equality guard, called directly -- same reasoning as Map
    Options' own version of this test: Qt already de-duplicates an
    identical setChecked/setCurrentIndex/setValue, so a widget-driven test
    would measure nothing."""
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(BLANK_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        for spec in panel._specs:
            panel._changed(spec, panel.current_values().get(spec.field_id))
        assert reported == [], f"a no-op change reported an edit: {reported}"
    finally:
        panel.deleteLater()


def test_changed_reports_nothing_while_populating() -> None:
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(BLANK_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        spec = next(s for s in panel._specs if s.field_id == "base_priority")
        target = panel.current_values()["base_priority"] + 1
        panel._populating = True
        panel._changed(spec, target)
        assert reported == []
        panel._populating = False
        panel._changed(spec, target)
        assert reported == [("base_priority", 1, target)]
    finally:
        panel.deleteLater()


def test_switching_player_reports_edits_for_the_right_player() -> None:
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(BLANK_FIXTURE)
    panel, reported = _editable_panel(loaded)
    try:
        spec = next(s for s in panel._specs if s.field_id == "base_priority")
        panel.player_combo.setCurrentIndex(2)  # P3
        panel._changed(spec, panel.current_values()["base_priority"] + 1)
        panel.player_combo.setCurrentIndex(4)  # P5
        panel._changed(spec, panel.current_values()["base_priority"] + 1)
        assert [player_id for _, player_id, _ in reported] == [3, 5]
    finally:
        panel.deleteLater()


def test_an_edit_round_trips_through_the_window_with_undo_redo() -> None:
    window = _players_window()
    try:
        panel = window.players_panel
        before = panel.current_values()["base_priority"]
        widget = panel.widget_for("base_priority")

        widget.setValue(before + 1)
        assert window.edit_history.is_dirty
        assert window.option_edits is not None
        assert window.option_edits.current_value("player:base_priority:1") == before + 1

        window.undo()
        assert panel.current_values()["base_priority"] == before
        assert not window.edit_history.is_dirty

        window.redo()
        assert panel.current_values()["base_priority"] == before + 1
    finally:
        _close(window)


def test_a_civilization_edit_round_trips_through_the_window_with_undo_redo() -> None:
    """Step B: civilization is editable on BLANK_FIXTURE (1.58, str16),
    same undo/redo shape as any other Players mode field despite riding
    OptionsEditModel.serialize_resizes() rather than serialize_patches()
    at save time -- that split is invisible above the model layer."""
    from PyQt5.QtWidgets import QComboBox

    window = _players_window()
    try:
        panel = window.players_panel
        widget = panel.widget_for("civilization")
        assert isinstance(widget, QComboBox)
        before = panel.current_values()["civilization"]
        target_index = next(i for i in range(widget.count()) if widget.itemData(i) != before)
        after = widget.itemData(target_index)

        widget.setCurrentIndex(target_index)
        assert window.edit_history.is_dirty
        assert window.option_edits is not None
        assert window.option_edits.current_value("player:civilization:1") == after

        window.undo()
        assert panel.current_values()["civilization"] == before
        assert not window.edit_history.is_dirty

        window.redo()
        assert panel.current_values()["civilization"] == after
    finally:
        _close(window)


def test_switching_players_preserves_pending_edits_on_the_other_player() -> None:
    """A P3 edit must still show when the combo returns to P3 after
    visiting P1, without a round trip back through the window -- see
    PlayersPanel.show_scenario()'s pending_values parameter."""
    from descape.player_fields import current_value

    window = _players_window()
    try:
        panel = window.players_panel
        spec = next(s for s in panel._specs if s.field_id == "base_priority")
        panel.player_combo.setCurrentIndex(2)  # P3
        widget = panel.widget_for("base_priority")
        before = panel.current_values()["base_priority"]
        widget.setValue(before + 7)

        panel.player_combo.setCurrentIndex(0)  # P1 -- unedited
        assert panel.current_values()["base_priority"] == current_value(window.scenario, spec, 1)

        panel.player_combo.setCurrentIndex(2)  # back to P3
        assert panel.current_values()["base_priority"] == before + 7
    finally:
        _close(window)


# --- corpus tier -----------------------------------------------------------


@pytest.mark.corpus
def test_players_panel_populates_every_corpus_file(scenario_path) -> None:
    """The tier that exercises the version-gated fields: 6 of 20 files
    predate per_player_population_cap (Pop Limit falls back to
    Units.player_data_4), 2 predate initial_player_views (Point of View
    group absent entirely), and 1 (the 1.37 file) has GAIA's
    0xFFFFFFFF starting age -- irrelevant here since the panel never shows
    player_id 0, but its presence is what makes P1..P8 on that same file
    worth checking for real."""
    from descape.player_fields import current_value

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Players")
        panel = window.players_panel
        assert panel._loaded is window.scenario
        assert panel.player_combo.count() == 8

        for player_id in (1, 8):
            panel.player_combo.setCurrentIndex(player_id - 1)
            for spec in panel._specs:
                raw = current_value(window.scenario, spec, player_id)
                shown = _shown_value(panel, spec)
                assert str(shown) == str(_expected_display(spec, raw)), (
                    f"{scenario_path.name} P{player_id} {spec.field_id}: "
                    f"shows {shown!r}, file holds {raw!r}"
                )
    finally:
        _close(window)


@pytest.mark.corpus
def test_browsing_players_leaves_the_document_clean(scenario_path, tmp_path) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Players")
        panel = window.players_panel
        for player_index in range(8):
            panel.player_combo.setCurrentIndex(player_index)
            for spec in panel._specs:
                panel.widget_for(spec.field_id).setFocus()
                QApplication.processEvents()
        assert not window.edit_history.is_dirty
        assert window.trigger_edits is None
        assert window.option_edits is None

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


# --- Number of Players (step 3e) ---------------------------------------
#
# The one row on this panel that is neither a PlayerFieldSpec nor
# per-player: a spinbox above the player selector, with its own gate. The
# byte-level half of these claims lives in
# tests/test_player_options_write_path.py; this covers the panel/window
# wiring.


def _count_panel(loaded, editable: bool = True):
    """A standalone panel with only Number of Players editable, recording
    what it reports -- same isolation _editable_panel() gives, scoped to
    the one row that has no spec behind it."""
    from descape.player_fields import PLAYER_COUNT_FIELD_ID
    from descape.players_panel import PlayersPanel

    conftest.ensure_qapp()
    reported = []
    panel = PlayersPanel(on_player_count=reported.append)
    panel.show_scenario(
        loaded,
        editable_fields=[PLAYER_COUNT_FIELD_ID] if editable else [],
        read_only_reasons={PLAYER_COUNT_FIELD_ID: "gate failed"},
    )
    return panel, reported


def test_the_count_spinbox_sits_above_the_player_selector() -> None:
    """Placement is the point, not decoration: inside a group box it would
    read as a setting of whichever player is selected."""
    window = _players_window()
    try:
        panel = window.players_panel
        layout = panel.layout()
        # The spinbox lives in a nested QHBoxLayout with its label, so find
        # which of the panel's own rows contains it.
        rows = [
            i
            for i in range(layout.count())
            if layout.itemAt(i).layout() is not None
            and layout.itemAt(i).layout().indexOf(panel.player_count_spin) >= 0
        ]
        assert rows, "the count row is not in the panel's own layout"
        assert rows[0] < layout.indexOf(panel.player_combo)
    finally:
        _close(window)


def test_the_count_shows_what_the_file_stores() -> None:
    window = _players_window()
    try:
        from descape.player_fields import defined_player_count

        panel = window.players_panel
        assert panel.current_player_count() == defined_player_count(window.scenario)
        assert panel.player_count_spin.value() == panel.current_player_count()
    finally:
        _close(window)


def test_the_count_is_editable_when_its_gate_holds() -> None:
    window = _players_window()
    try:
        from descape.player_fields import PLAYER_COUNT_FIELD_ID

        assert PLAYER_COUNT_FIELD_ID in window._editable_player_fields()
        assert window.players_panel.player_count_spin.isEnabled()
    finally:
        _close(window)


def test_a_read_only_count_is_disabled_and_carries_its_gate_reason() -> None:
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(BLANK_FIXTURE)
    panel, reported = _count_panel(loaded, editable=False)
    try:
        assert not panel.player_count_spin.isEnabled()
        assert panel.player_count_spin.toolTip() == "gate failed"
        panel.player_count_spin.setValue(panel.current_player_count() + 1)
        assert reported == []
    finally:
        panel.deleteLater()


def test_the_count_reports_a_real_change_once() -> None:
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(BLANK_FIXTURE)
    panel, reported = _count_panel(loaded)
    try:
        before = panel.current_player_count()
        panel.player_count_spin.setValue(before + 1)
        assert reported == [before + 1]
        panel.player_count_spin.setValue(before + 1)  # same value again -- a no-op
        assert reported == [before + 1]
    finally:
        panel.deleteLater()


def test_populating_the_count_reports_nothing() -> None:
    from descape.scenario_io import load_map_and_units

    from descape.player_fields import PLAYER_COUNT_FIELD_ID

    loaded = load_map_and_units(BLANK_FIXTURE)
    panel, reported = _count_panel(loaded)
    try:
        panel.show_scenario(
            loaded, editable_fields=[PLAYER_COUNT_FIELD_ID], player_count=7
        )
        assert panel.current_player_count() == 7
        assert reported == []
    finally:
        panel.deleteLater()


def test_a_count_edit_round_trips_through_the_window_with_undo_redo() -> None:
    window = _players_window()
    try:
        from descape.player_fields import PLAYER_COUNT_FIELD_ID

        panel = window.players_panel
        before = panel.current_player_count()

        panel.player_count_spin.setValue(before + 1)
        assert window.edit_history.is_dirty
        assert window.option_edits is not None
        assert window.option_edits.current_value(PLAYER_COUNT_FIELD_ID) == before + 1

        window.undo()
        assert panel.current_player_count() == before
        assert panel.player_count_spin.value() == before
        assert not window.edit_history.is_dirty

        window.redo()
        assert panel.current_player_count() == before + 1
        assert panel.player_count_spin.value() == before + 1
    finally:
        _close(window)


def test_a_count_edit_resizes_the_diplomacy_grid() -> None:
    """Diplomacy mode's grid was built on the premise that nothing ever
    edits the active-player set. That ends here, so the grid has to react
    to a pending count in the same session, not only after a save and
    reload."""
    window = _players_window()
    try:
        panel = window.players_panel
        before = panel.current_player_count()
        panel.player_count_spin.setValue(before + 2)

        window.mode_combo.setCurrentText("Diplomacy")
        assert window.diplomacy_panel._active_players == list(range(1, before + 3))
        assert window.diplomacy_panel.player_combo.count() == before + 2
    finally:
        _close(window)


def test_browsing_the_count_leaves_the_document_clean() -> None:
    """Same acceptance gate every other row here has: focusing and reading
    the spinbox must never dirty the document."""
    from PyQt5.QtWidgets import QApplication

    window = _players_window()
    try:
        window.players_panel.player_count_spin.setFocus()
        QApplication.processEvents()
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_the_status_line_does_not_count_the_count_row_as_a_setting() -> None:
    """Regression, caught by looking at an offscreen render rather than by
    an assertion: the editable set carries PLAYER_COUNT_FIELD_ID, which has
    no PlayerFieldSpec, so `total - len(editable_fields)` undercounted the
    read-only rows by one (and reported a negative count on a synthetic set
    where every spec was editable)."""
    from descape.player_fields import _NEVER_WRITABLE

    window = _players_window()
    try:
        panel = window.players_panel
        expected = sum(1 for s in panel._specs if s.field_id not in panel._editable_fields)
        assert expected == len(_NEVER_WRITABLE)
        assert f", {expected} read-only." in panel.status.text(), panel.status.text()
    finally:
        _close(window)
