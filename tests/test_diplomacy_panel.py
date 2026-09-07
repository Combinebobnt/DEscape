"""Coverage for Diplomacy mode and DiplomacyPanel: a player selector above a
form of stances and allied-victory flags, writable through the same
OptionsEditModel Map Options and Players mode ride (step 4).

Same offscreen-ViewerWindow technique as tests/test_players_panel.py, and
this panel is closest in shape to that one: a player selector above a form,
`editable_fields` resolved by the window from two independent gates
(options_write_supported carries construction, diplomacy_write_supported
gates the grid specifically -- see viewer.py's _editable_diplomacy_fields()),
"browsing never dirties the document" alongside "an edit round-trips".

Two differences from PlayersPanel drive most of the tests here:

- The selector lists only this file's *defined* players
  (player_fields.defined_player_count()), not a fixed P1..P8, so a
  synthetic 5-defined-player scenario is built by flipping
  DataHeader.player_data_1[].active in the already-parsed retriever data --
  the panel reads player counts and stances off the parsed structures, not
  raw bytes, so this is a direct and minimal way to get a non-default
  defined_player_count without depending on the untracked examples/ corpus
  for a default-tier test.
- The self cell (stance[i][i]) is real per-file data with no row of its own,
  so it is covered by status-text tests instead of a widget lookup.
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


def _diplomacy_window(path=BLANK_FIXTURE):
    window = _window()
    window.load_scenario(path)
    window.mode_combo.setCurrentText("Diplomacy")
    return window


def _set_defined_player_count(loaded, count: int) -> None:
    """Flip DataHeader.player_data_1[].active on an already-loaded scenario
    so defined_player_count(loaded) reports `count` -- see module docstring
    for why this, rather than a corpus file, is this suite's source of a
    non-default player count."""
    player_data_1 = loaded._scenario.sections["DataHeader"].retriever_map["player_data_1"].data
    for i, entry in enumerate(player_data_1[:8]):
        entry.retriever_map["active"].data = 1 if i < count else 0


def _set_active_players(loaded, player_ids) -> None:
    """Like _set_defined_player_count, but for an arbitrary (possibly
    non-contiguous) set of player ids -- what select_player()'s sparse-set
    coverage below needs, since a contiguous 1..N active run can't catch a
    regression to the `index + 1` arithmetic PlayersPanel uses."""
    player_data_1 = loaded._scenario.sections["DataHeader"].retriever_map["player_data_1"].data
    active = set(player_ids)
    for i, entry in enumerate(player_data_1[:8]):
        entry.retriever_map["active"].data = 1 if (i + 1) in active else 0


def _set_stance(loaded, row_player: int, col_player: int, value: int) -> None:
    row_struct = loaded._scenario.sections["Diplomacy"].retriever_map["per_player_diplomacy"].data[
        row_player - 1
    ]
    row_struct.retriever_map["stance_with_each_player"].data[col_player - 1] = value


# --- the mode ----------------------------------------------------------


def test_diplomacy_mode_swaps_the_left_panel_and_gates_edit_tools() -> None:
    from descape.viewer import _LEFT_PAGE_DIPLOMACY

    window = _window()
    try:
        assert window.left_stack.currentIndex() == 0
        window.mode_combo.setCurrentText("Diplomacy")
        assert window.mode == "diplomacy"
        assert window.left_stack.currentIndex() == _LEFT_PAGE_DIPLOMACY
        assert not window.draw_action.isEnabled()
        assert not window.elevation_action.isEnabled()
        assert window.pan_action.isChecked()

        window.mode_combo.setCurrentText("View")
        assert window.left_stack.currentIndex() == 0
    finally:
        _close(window)


def test_diplomacy_mode_appended_to_the_combo() -> None:
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
        window.mode_combo.setCurrentText("Diplomacy")
        panel = window.diplomacy_panel
        assert panel.status.text() == panel._NO_DOCUMENT
        assert not panel.player_combo.isEnabled()
    finally:
        _close(window)


def test_closing_the_document_clears_the_panel() -> None:
    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        assert panel.player_combo.isEnabled()
        window.close_scenario()
        assert panel.status.text() == panel._NO_DOCUMENT
        assert not panel.player_combo.isEnabled()
        assert panel.player_combo.count() == 0
    finally:
        _close(window)


def test_opening_a_second_map_while_in_the_mode_repopulates() -> None:
    """Its own if/else in load_scenario, not chained onto Map Options' or
    Players' -- see the plan's touch list. Opening a second file while
    already in Diplomacy mode must show the new file's data."""
    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        first_status = panel.status.text()

        window.load_scenario(BLANK_FIXTURE)
        assert panel._loaded is window.scenario
        assert panel.status.text() == first_status  # same fixture -- sanity check
    finally:
        _close(window)


# --- the form: defined-player-count-only selector -----------------------


def test_player_selector_lists_only_defined_players() -> None:
    """The blank template defines exactly 2 players (pinned by
    test_player_write_path.test_defined_player_count_matches_the_known_blank_template_default)
    -- the selector must not fall back to a fixed P1..P8 the way
    PlayersPanel's does."""
    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        assert panel.player_combo.count() == 2
        labels = [panel.player_combo.itemText(i) for i in range(2)]
        assert labels == ["P1", "P2"]
        for i in range(2):
            assert not panel.player_combo.itemIcon(i).isNull()
        assert panel.player_combo.currentIndex() == 0
    finally:
        _close(window)


def test_select_player_happy_path() -> None:
    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        assert panel.select_player(2) is True
        assert panel.player_combo.currentIndex() == 1

        # Already-current player is a no-op that still reports success.
        assert panel.select_player(2) is True
        assert panel.player_combo.currentIndex() == 1
    finally:
        _close(window)


def test_select_player_rejects_gaia() -> None:
    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        index_before = panel.player_combo.currentIndex()
        assert panel.select_player(0) is False
        assert panel.player_combo.currentIndex() == index_before
    finally:
        _close(window)


def test_select_player_on_sparse_active_set_resolves_by_player_id_not_index() -> None:
    """Without this, a regression to `index + 1` (the arithmetic
    PlayersPanel uses, wrong here since the active set isn't contiguous
    from 1) would pass every other test in this file and still resolve
    combo index N to the wrong player."""
    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        _set_active_players(window.scenario, [1, 3, 5])
        panel.show_scenario(window.scenario)
        assert [panel.player_combo.itemText(i) for i in range(3)] == ["P1", "P3", "P5"]

        assert panel.select_player(3) is True
        assert panel.player_combo.currentIndex() == 1

        assert panel.select_player(2) is False
    finally:
        _close(window)


def test_rows_are_defined_player_count_minus_one() -> None:
    """Rows are every other defined player, so the count is
    defined_player_count(loaded) - 1. Checked against a synthetic 5-player
    file so this does not pass vacuously against the blank template's
    2-player, 1-row case."""
    from descape.diplomacy_fields import stance_cell_id

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        _set_defined_player_count(window.scenario, 5)
        panel.show_scenario(window.scenario)
        assert panel.player_combo.count() == 5

        for player_id in range(1, 6):
            panel.player_combo.setCurrentIndex(player_id - 1)
            expected = {stance_cell_id(player_id, o) for o in range(1, 6) if o != player_id}
            allied_ids = {k for k in panel._widgets if k.startswith("allied_victory:")}
            assert set(panel._widgets) - allied_ids == expected
            assert len(expected) == 4  # 5 - 1
    finally:
        _close(window)


def test_toward_labels_name_the_real_player_number_not_a_relative_index() -> None:
    """The plan's own mockup: with Player 3 selected, the rows read 'Toward
    Player 1', 'Toward Player 2', 'Toward Player 4' -- skipping 3, not
    relabelled 1/2/3."""
    from PyQt5.QtWidgets import QFormLayout

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        _set_defined_player_count(window.scenario, 5)
        panel.show_scenario(window.scenario)
        panel.player_combo.setCurrentIndex(2)  # Player 3

        box = panel.host.findChild(__import__("PyQt5.QtWidgets", fromlist=["QGroupBox"]).QGroupBox)
        form = box.layout()
        labels = []
        for i in range(form.rowCount()):
            item = form.itemAt(i, QFormLayout.LabelRole)
            if item:
                labels.append(item.widget().text())
        assert labels == [
            "Toward Player 1",
            "Toward Player 2",
            "Toward Player 4",
            "Toward Player 5",
            "Allied victory",
        ]
    finally:
        _close(window)


def test_switching_player_repopulates_the_form_with_that_players_stances() -> None:
    from descape.diplomacy_fields import allied_victory_cell_id, allied_victory_value, stance_cell_id, stance_value

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        for player_id in (1, 2):
            panel.player_combo.setCurrentIndex(player_id - 1)
            assert panel._player_id == player_id
            other = 2 if player_id == 1 else 1
            widget = panel.widget_for(stance_cell_id(player_id, other))
            assert widget.button_group.checkedId() == stance_value(window.scenario, player_id, other)
            allied_widget = panel.widget_for(allied_victory_cell_id(player_id))
            assert allied_widget.isChecked() == bool(allied_victory_value(window.scenario, player_id))
    finally:
        _close(window)


def _is_grid_cell(field_id: str) -> bool:
    """stance:i:j / allied_victory:i, as opposed to a bare Teams-group spec
    field id like "lock_teams" -- see module docstring's collision note."""
    return field_id.startswith("stance:") or field_id.startswith("allied_victory:")


def _row_enabled(widget) -> bool:
    """True iff every interactive control in this row is enabled -- a
    stance row's QRadioButtons, or a Teams-group QCheckBox/QSpinBox
    directly."""
    from PyQt5.QtWidgets import QCheckBox, QRadioButton, QSpinBox

    if isinstance(widget, (QCheckBox, QSpinBox)):
        return widget.isEnabled()
    radios = widget.findChildren(QRadioButton)
    assert radios, "not a recognized row widget kind"
    return all(r.isEnabled() for r in radios)


def test_every_row_is_enabled_when_the_write_path_is_available() -> None:
    """The blank template passes both gates, so step 4 (grid cells) and
    step 5 (the Teams group) land every row enabled -- the counterpart to
    the gate-failure test below."""
    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        assert panel._widgets, "no rows built -- this would pass vacuously"
        for field_id, widget in panel._widgets.items():
            assert _row_enabled(widget), field_id
    finally:
        _close(window)


def test_grid_rows_are_disabled_with_a_reason_when_the_diplomacy_gate_fails() -> None:
    """Corrupting one stance byte fails diplomacy_write_supported() without
    touching options_write_supported() -- same technique
    tests/test_diplomacy_write_path.py's own gate-failure test uses. Every
    grid row must grey out with a tooltip explaining why, not just the
    corrupted cell -- the gate is document-wide, not per-cell (see
    viewer.py._editable_diplomacy_fields()). The Teams group is a separate,
    independent gate (options_write_supported), so it stays enabled here --
    that independence is the point of the two-gate design, and is asserted
    below rather than left implicit."""
    from descape.diplomacy_fields import stance_cell_id, stance_offsets

    window = _window()
    try:
        window.load_scenario(BLANK_FIXTURE)
        target = stance_offsets(window.scenario)[stance_cell_id(3, 5)]
        body = bytearray(window.scenario.decompressed_body)
        body[target.offset] ^= 0xFF
        window.scenario.decompressed_body = bytes(body)

        window.mode_combo.setCurrentText("Diplomacy")
        panel = window.diplomacy_panel
        grid_widgets = {fid: w for fid, w in panel._widgets.items() if _is_grid_cell(fid)}
        team_widgets = {fid: w for fid, w in panel._widgets.items() if not _is_grid_cell(fid)}
        assert grid_widgets, "no grid rows built -- this would pass vacuously"
        assert team_widgets, "no Teams rows built -- this would pass vacuously"
        for field_id, widget in grid_widgets.items():
            assert not _row_enabled(widget), field_id
            assert widget.toolTip(), f"{field_id} is disabled with no explanation"
        for field_id, widget in team_widgets.items():
            assert _row_enabled(widget), field_id
    finally:
        _close(window)


def test_populating_the_panel_records_no_undo_step() -> None:
    """radio.setChecked(True) inside _build_stance_row fires toggled()
    during every populate -- without the panel's own `_populating` guard
    this would record one phantom OptionsDiffRecord per row on every mode
    entry, undo, and player switch."""
    window = _diplomacy_window()
    try:
        assert not window.edit_history.records
        assert not window.edit_history.is_dirty
        assert window.option_edits is None
        window.diplomacy_panel.player_combo.setCurrentIndex(1)
        assert not window.edit_history.records
    finally:
        _close(window)


def test_an_out_of_enum_stance_is_shown_as_stored_not_silently_unchecked() -> None:
    """Only {0,1,3} appear anywhere in the grid across the corpus, but
    nothing here should leave every radio unchecked for a value none of the
    three represent -- the same rule MapOptionsPanel's own as-stored label
    follows."""
    from PyQt5.QtWidgets import QLabel

    window = _diplomacy_window()
    try:
        _set_stance(window.scenario, 1, 2, 7)
        panel = window.diplomacy_panel
        panel.show_scenario(window.scenario)

        from descape.diplomacy_fields import stance_cell_id

        widget = panel.widget_for(stance_cell_id(1, 2))
        assert widget.button_group.checkedId() == -1
        note = widget.findChild(QLabel)
        assert note is not None and note.text() == "stored value 7"
    finally:
        _close(window)


# --- the self-cell note ---------------------------------------------------


def test_a_default_self_stance_gets_no_note() -> None:
    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        assert "Self-stance" not in panel.status.text()
    finally:
        _close(window)


def test_a_non_default_self_stance_gets_a_per_player_note() -> None:
    """Per selected player, not per file -- switching away from the mutated
    player must drop the note, and back to it must show it again."""
    window = _diplomacy_window()
    try:
        _set_stance(window.scenario, 1, 1, 0)  # ALLY, not the default ENEMY (3)
        panel = window.diplomacy_panel
        panel.show_scenario(window.scenario)

        assert "Self-stance stored as Ally" in panel.status.text()
        assert "Player 1" in panel.status.toolTip()

        panel.player_combo.setCurrentIndex(1)  # Player 2, untouched
        assert "Self-stance" not in panel.status.text()

        panel.player_combo.setCurrentIndex(0)  # back to Player 1
        assert "Self-stance stored as Ally" in panel.status.text()
    finally:
        _close(window)


def test_self_note_is_not_counted_as_a_row_or_a_widget() -> None:
    """'Do not route it through an as-stored count': the self cell has no
    row and no entry in _widgets, since MapOptionsPanel._status_text()'s
    as-stored count works by scanning _widgets for a QLabel."""
    window = _diplomacy_window()
    try:
        _set_stance(window.scenario, 1, 1, 0)
        panel = window.diplomacy_panel
        panel.show_scenario(window.scenario)
        for field_id in panel._widgets:
            assert field_id not in ("stance:1:1",)
    finally:
        _close(window)


# --- geometry --------------------------------------------------------------


def test_every_widget_fits_min_useful_width() -> None:
    from descape.diplomacy_panel import DiplomacyPanel

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        _set_defined_player_count(window.scenario, 8)
        panel.show_scenario(window.scenario)
        for field_id, widget in panel._widgets.items():
            width = widget.minimumSizeHint().width()
            assert width <= DiplomacyPanel.MIN_USEFUL_WIDTH, f"{field_id} demands {width} px"
    finally:
        _close(window)


def test_every_group_renders_at_the_height_its_layout_asks_for() -> None:
    """The plan's own acceptance gate 7: 'an offscreen render asserting no
    row demands more width than the pane gives and every group renders at
    its heightForWidth()' -- 7 radio rows of 3 buttons is the widest thing
    this codebase has put in this pane. Same technique as
    test_map_options_panel.py's own copy of this test."""
    from PyQt5.QtWidgets import QApplication, QGroupBox

    from descape.diplomacy_panel import DiplomacyPanel

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        _set_defined_player_count(window.scenario, 8)
        panel.show_scenario(window.scenario)
        panel.resize(DiplomacyPanel.MIN_USEFUL_WIDTH, 760)
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
                f"{field_id} is {widget.height()} px, needs {widget.minimumSizeHint().height()}"
            )
    finally:
        _close(window)


# --- read-only containment -----------------------------------------------


def test_browsing_every_row_leaves_the_document_clean() -> None:
    """Calling show_scenario() directly with no editable_fields (as this
    forces) is the panel's own read-only construction path -- there is no
    callback for a disabled widget to fire in the first place, but this is
    the acceptance gate that keeps it that way."""
    from PyQt5.QtWidgets import QApplication, QRadioButton

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        _set_defined_player_count(window.scenario, 8)
        panel.show_scenario(window.scenario)
        for player_index in range(8):
            panel.player_combo.setCurrentIndex(player_index)
            for widget in panel._widgets.values():
                for radio in widget.findChildren(QRadioButton) or [widget]:
                    radio.setFocus()
                    QApplication.processEvents()
        assert not window.edit_history.is_dirty
        assert window.option_edits is None
    finally:
        _close(window)


def test_focusing_enabled_rows_without_toggling_leaves_the_document_clean() -> None:
    """The counterpart with the write path actually available (through the
    window, unlike the test above): every row is enabled, and merely
    focusing it must still not dirty the document -- only an actual toggle
    may."""
    from PyQt5.QtWidgets import QApplication, QRadioButton

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        for widget in panel._widgets.values():
            for radio in widget.findChildren(QRadioButton) or [widget]:
                assert radio.isEnabled()
                radio.setFocus()
                QApplication.processEvents()
        assert not window.edit_history.is_dirty
        assert window.option_edits is None
    finally:
        _close(window)


# --- editing, through the window --------------------------------------------


def test_clicking_a_stance_radio_records_one_undo_step_and_dirties_the_document() -> None:
    from descape.diplomacy_fields import stance_cell_id

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        cell_id = stance_cell_id(1, 2)
        widget = panel.widget_for(cell_id)
        stored = widget.button_group.checkedId()
        other = next(v for v in (0, 1, 3) if v != stored)
        widget.button_group.button(other).setChecked(True)

        assert window.option_edits is not None, "the edit built no options model"
        assert window.option_edits.current_value(cell_id) == other
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[0].kind == "options"
        assert window.edit_history.is_dirty
        assert window.trigger_edits is None
    finally:
        _close(window)


def test_clicking_the_allied_victory_checkbox_records_one_undo_step() -> None:
    from descape.diplomacy_fields import allied_victory_cell_id

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        cell_id = allied_victory_cell_id(1)
        widget = panel.widget_for(cell_id)
        before = widget.isChecked()
        widget.setChecked(not before)

        assert window.option_edits is not None
        assert window.option_edits.current_value(cell_id) == int(not before)
        assert len(window.edit_history.records) == 1
        assert window.edit_history.is_dirty
    finally:
        _close(window)


def test_undoing_a_stance_edit_puts_the_row_back_and_keeps_the_selected_player() -> None:
    """The repopulate matters as much as the model restore (same reasoning
    as test_map_options_panel.py's own version of this test), and so does
    the selected player: an undo must not silently snap the panel back to
    P1 while the user is looking at a different one."""
    from descape.diplomacy_fields import stance_cell_id

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        panel.player_combo.setCurrentIndex(1)  # P2
        cell_id = stance_cell_id(2, 1)
        widget = panel.widget_for(cell_id)
        stored = widget.button_group.checkedId()
        other = next(v for v in (0, 1, 3) if v != stored)
        widget.button_group.button(other).setChecked(True)

        window.undo()

        assert panel._player_id == 2, "undo silently changed the selected player"
        assert panel.current_values()[cell_id] == stored
        assert panel.widget_for(cell_id).button_group.checkedId() == stored
        assert not window.option_edits.has_edits
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_an_undone_and_redone_stance_edit_ends_where_it_started() -> None:
    from descape.diplomacy_fields import stance_cell_id

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        cell_id = stance_cell_id(1, 2)
        widget = panel.widget_for(cell_id)
        stored = widget.button_group.checkedId()
        other = next(v for v in (0, 1, 3) if v != stored)
        widget.button_group.button(other).setChecked(True)

        window.undo()
        window.redo()

        assert panel.current_values()[cell_id] == other
        assert window.option_edits.current_value(cell_id) == other
        assert window.edit_history.is_dirty
    finally:
        _close(window)


def test_switching_player_preserves_a_pending_stance_edit_on_the_other_player() -> None:
    """A P1-toward-P2 edit must still show when the combo returns to P1
    after visiting P2, without a round trip back through the window -- see
    DiplomacyPanel.show_scenario()'s pending_values parameter and
    PlayersPanel's own version of this test."""
    from descape.diplomacy_fields import stance_cell_id, stance_value

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        cell_id = stance_cell_id(1, 2)
        widget = panel.widget_for(cell_id)
        stored = widget.button_group.checkedId()
        other = next(v for v in (0, 1, 3) if v != stored)
        widget.button_group.button(other).setChecked(True)

        panel.player_combo.setCurrentIndex(1)  # P2 -- the reverse direction, untouched
        assert panel.current_values()[stance_cell_id(2, 1)] == stance_value(
            window.scenario, 2, 1
        )

        panel.player_combo.setCurrentIndex(0)  # back to P1
        assert panel.current_values()[cell_id] == other
        assert panel.widget_for(cell_id).button_group.checkedId() == other
    finally:
        _close(window)


# --- the Teams group (step 5) -----------------------------------------------
#
# lock_teams, allow_players_choose_teams, random_start_points and
# max_number_of_teams moved here from Map Options mode (step 5). They are
# ordinary option_fields.OptionFieldSpec rows (not grid cells), reported through the
# panel's second callback (on_option_field), routed to the same
# ViewerWindow.set_option_field() every other Map Options row still uses.


def test_the_teams_group_lists_all_four_specs_in_order() -> None:
    from PyQt5.QtWidgets import QFormLayout, QGroupBox

    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        box = next(b for b in panel.host.findChildren(QGroupBox) if b.title() == "Teams")
        form = box.layout()
        labels = []
        for i in range(form.rowCount()):
            item = form.itemAt(i, QFormLayout.LabelRole)
            if item:
                labels.append(item.widget().text())
        assert labels == [
            "Lock teams",
            "Players choose teams",
            "Random start points",
            "Max number of teams",
        ]
    finally:
        _close(window)


def test_clicking_the_lock_teams_checkbox_records_one_undo_step_and_dirties_the_document() -> None:
    """The Teams group's counterpart to the stance/allied-victory edit tests
    above -- same OptionsEditModel, same _option_edit() wrapper, reached
    through on_option_field instead of on_diplomacy_field."""
    window = _diplomacy_window()
    try:
        panel = window.diplomacy_panel
        widget = panel.widget_for("lock_teams")
        before = widget.isChecked()
        widget.setChecked(not before)

        assert window.option_edits is not None
        assert window.option_edits.current_value("lock_teams") == int(not before)
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[0].kind == "options"
        assert window.edit_history.is_dirty

        window.undo()
        assert panel.widget_for("lock_teams").isChecked() == before
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_a_read_only_reason_wins_over_random_start_points_own_tooltip() -> None:
    """random_start_points carries a spec tooltip ("No observed effect...")
    and can also be greyed by a gate -- why the row cannot be edited at all
    is the more useful of the two. The Map Options collision test this moved
    from used a bare panel with an explicit read_only_reasons override;
    that construction still works here since option_specs/editable_fields/
    read_only_reasons are passed the same way MapOptionsPanel's are."""
    from descape import option_fields
    from descape.diplomacy_panel import DiplomacyPanel
    from descape.scenario_io import load_map_and_units

    conftest.ensure_qapp()
    loaded = load_map_and_units(BLANK_FIXTURE)
    specs = [s for s in option_fields.specs_for(loaded) if s.panel == "diplomacy"]
    panel = DiplomacyPanel()
    try:
        panel.show_scenario(loaded, option_specs=specs, editable_fields=())
        assert "No observed effect" in panel.widget_for("random_start_points").toolTip()

        panel.show_scenario(
            loaded,
            option_specs=specs,
            editable_fields=(),
            read_only_reasons={"random_start_points": "a gate refused this row"},
        )
        assert panel.widget_for("random_start_points").toolTip() == "a gate refused this row"
    finally:
        panel.deleteLater()


def test_a_spec_tooltip_never_overwrites_an_out_of_range_explanation() -> None:
    """random_start_points is the live collision, moved here from Map
    Options' own version of this test: it is a checkbox (so it can take the
    out-of-range branch) and it carries its own spec-level tooltip. The
    row's bare number has no other explanation anywhere, so the spec note
    is the one that gives way."""
    from PyQt5.QtWidgets import QLabel

    from descape import option_fields
    from descape.diplomacy_panel import DiplomacyPanel
    from descape.scenario_io import load_map_and_units

    conftest.ensure_qapp()
    specs = [s for s in option_fields._SPECS if s.panel == "diplomacy"]
    spec = next(s for s in specs if s.field_id == "random_start_points")
    assert spec.tooltip, "this test needs a spec that carries its own tooltip"

    loaded = load_map_and_units(BLANK_FIXTURE)
    parsed = loaded._scenario.sections["Diplomacy"].retriever_map["random_start_points"]
    original = parsed.data
    panel = DiplomacyPanel()
    try:
        parsed.data = 7
        panel.show_scenario(loaded, option_specs=specs)
        widget = panel.widget_for("random_start_points")
        assert isinstance(widget, QLabel)
        assert "editable range" in widget.toolTip()
        assert widget.toolTip() != spec.tooltip

        # In range, the spec's own tooltip is still the one shown.
        parsed.data = original
        panel.show_scenario(loaded, option_specs=specs)
        assert panel.widget_for("random_start_points").toolTip() == spec.tooltip
    finally:
        parsed.data = original
        panel.deleteLater()


def test_max_number_of_teams_past_int32_is_shown_as_stored() -> None:
    """The past-int32-spinbox case moved from Map Options' own version of
    this test (test_a_value_its_editor_could_not_show_truthfully_is_shown_
    as_stored): a QSpinBox raises OverflowError on a value past int32
    outright, so it must render as a label instead. max_number_of_teams
    itself never carries such a value in any real file -- it arrives here
    through pending_values, the same way the Map Options version used
    show_scenario()'s values= override, to exercise the general mechanism
    against a field that would otherwise never take this branch."""
    from PyQt5.QtWidgets import QLabel

    from descape import option_fields
    from descape.diplomacy_panel import DiplomacyPanel
    from descape.scenario_io import load_map_and_units

    conftest.ensure_qapp()
    loaded = load_map_and_units(BLANK_FIXTURE)
    specs = [s for s in option_fields.specs_for(loaded) if s.panel == "diplomacy"]
    panel = DiplomacyPanel()
    try:
        panel.show_scenario(loaded, option_specs=specs, pending_values={"max_number_of_teams": 0xFFFFFFFF})
        widget = panel.widget_for("max_number_of_teams")
        assert isinstance(widget, QLabel), f"built a {type(widget).__name__}, not a label"
        assert widget.text() == str(0xFFFFFFFF)
        assert "editable range" in widget.toolTip()
        assert panel.current_values()["max_number_of_teams"] == 0xFFFFFFFF

        # The file's own in-range value on the same row still gets a real
        # editor, so this is a per-value decision rather than a dead field.
        panel.show_scenario(loaded, option_specs=specs)
        assert not isinstance(panel.widget_for("max_number_of_teams"), QLabel)
    finally:
        panel.deleteLater()


# --- corpus tier -----------------------------------------------------------


@pytest.mark.corpus
def test_diplomacy_panel_populates_every_corpus_file(scenario_path) -> None:
    from descape.diplomacy_fields import (
        allied_victory_cell_id,
        allied_victory_value,
        stance_cell_id,
        stance_value,
    )
    from descape.player_fields import defined_player_count

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Diplomacy")
        panel = window.diplomacy_panel
        assert panel._loaded is window.scenario
        count = defined_player_count(window.scenario)
        assert panel.player_combo.count() == count

        for player_id in range(1, count + 1):
            panel.player_combo.setCurrentIndex(player_id - 1)
            for opponent in range(1, count + 1):
                if opponent == player_id:
                    continue
                widget = panel.widget_for(stance_cell_id(player_id, opponent))
                raw = stance_value(window.scenario, player_id, opponent)
                assert widget.button_group.checkedId() == raw or (
                    widget.button_group.checkedId() == -1 and raw not in (0, 1, 3)
                ), (scenario_path.name, player_id, opponent)
            allied_widget = panel.widget_for(allied_victory_cell_id(player_id))
            allied_raw = allied_victory_value(window.scenario, player_id)
            assert allied_widget.isChecked() == bool(allied_raw), (scenario_path.name, player_id)
    finally:
        _close(window)


@pytest.mark.corpus
def test_the_teams_group_populates_every_corpus_file(scenario_path) -> None:
    """The corpus coverage test_map_options_panel_populates_every_corpus_file
    used to give these four fields before step 5 moved them here -- the
    section is byte-identical across every DE structure version measured,
    so unlike the rest of Map Options they are never version-gated absent."""
    from PyQt5.QtWidgets import QCheckBox, QLabel, QSpinBox

    from descape import option_fields

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Diplomacy")
        panel = window.diplomacy_panel
        specs = [s for s in option_fields.specs_for(window.scenario) if s.panel == "diplomacy"]
        assert len(specs) == 4, (scenario_path.name, len(specs))
        for spec in specs:
            widget = panel.widget_for(spec.field_id)
            raw = int(option_fields.current_value(window.scenario, spec))
            if isinstance(widget, QLabel):
                assert widget.text() == str(raw), (scenario_path.name, spec.field_id)
            elif isinstance(widget, QCheckBox):
                assert widget.isChecked() == bool(raw), (scenario_path.name, spec.field_id)
            else:
                assert isinstance(widget, QSpinBox), (scenario_path.name, spec.field_id)
                assert widget.value() == raw, (scenario_path.name, spec.field_id)
    finally:
        _close(window)


@pytest.mark.corpus
def test_browsing_diplomacy_leaves_the_document_clean(scenario_path) -> None:
    from PyQt5.QtWidgets import QApplication, QRadioButton

    from descape.player_fields import defined_player_count

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Diplomacy")
        panel = window.diplomacy_panel
        count = defined_player_count(window.scenario)
        for player_index in range(count):
            panel.player_combo.setCurrentIndex(player_index)
            for widget in panel._widgets.values():
                for radio in widget.findChildren(QRadioButton) or [widget]:
                    radio.setFocus()
                    QApplication.processEvents()
        assert not window.edit_history.is_dirty
        assert window.trigger_edits is None
        assert window.option_edits is None
    finally:
        _close(window)
