"""DisablesDialog (GH #57) and its two entry points.

Same offscreen-ViewerWindow technique as tests/test_players_panel.py. The
dialog is driven through ViewerWindow._build_disables_dialog(), never
_show_disables_dialog(): exec_() is modal and would hang an offscreen run
outright, the same trap tests/test_keybinds.py's docstring records for
SettingsDialog.

The acceptance gates, in the order they matter:

- a whole dialog session is ONE undo step, because the dialog is modal and
  Ctrl+Z cannot reach the user while it is open;
- Cancel leaves the document clean -- a model dirty while the history is not
  closes the document with no save prompt;
- an id the Full List cannot produce still renders with a real name and
  survives an OK that never touched it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

_close = conftest.close_window


def _window(path=None):
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.shown_window(1200, 800)
    window.load_scenario(path or BLANK_TEMPLATE_PATH)
    assert window.scenario is not None
    return window


# -- entry points ------------------------------------------------------------


def test_both_entry_points_are_enabled_for_a_file_that_verifies() -> None:
    window = _window()
    try:
        assert window._disables_editable()
        assert window.disables_action.isEnabled()
        window.mode_combo.setCurrentText("Players")
        assert window.players_panel.disables_button.isEnabled()
    finally:
        _close(window)


def test_both_entry_points_are_disabled_with_no_document() -> None:
    window = conftest.shown_window(1200, 800)
    try:
        assert not window._disables_editable()
        assert not window.disables_action.isEnabled()
        assert not window.players_panel.disables_button.isEnabled()
        assert window._build_disables_dialog() is None
    finally:
        _close(window)


def test_a_file_whose_gate_fails_leaves_the_button_disabled_with_a_tooltip(monkeypatch) -> None:
    from descape import viewer as viewer_module
    from descape.players_panel import PlayersPanel

    window = _window()
    try:
        monkeypatch.setattr(viewer_module, "disables_write_supported", lambda *_: False)
        window.mode_combo.setCurrentText("Players")
        window._repopulate_players()
        window._update_tool_enabled()
        button = window.players_panel.disables_button
        assert not button.isEnabled()
        assert button.toolTip() == PlayersPanel._DISABLES_GATE_REASON
        assert not window.disables_action.isEnabled()
        assert window._build_disables_dialog() is None
    finally:
        _close(window)


def test_the_menu_action_is_registered_as_a_rebindable_keybind() -> None:
    from descape import settings

    window = conftest.shown_window(1200, 800)
    try:
        assert window._keybind_actions["edit_disables"] is window.disables_action
        assert ("edit_disables", "Disabled Objects…", "") in settings.REBINDABLE_ACTIONS
    finally:
        _close(window)


# -- the dialog itself -------------------------------------------------------


def test_ok_after_several_adds_across_two_categories_pushes_exactly_one_record() -> None:
    window = _window()
    try:
        before_depth = len(window.edit_history.records)
        dialog = window._build_disables_dialog()
        dialog.select_player(2)
        dialog.add_id("buildings", 109)
        dialog.add_id("buildings", 68)
        dialog.select_player(5)
        dialog.add_id("techs", 22)
        dialog._accept()

        assert len(window.edit_history.records) == before_depth + 1
        model = window.option_edits
        assert model.current_value("disabled:buildings:2") == (109, 68)
        assert model.current_value("disabled:techs:5") == (22,)
        assert model.has_disables_edits

        window.undo()
        assert model.current_value("disabled:buildings:2") == ()
        assert model.current_value("disabled:techs:5") == ()
        assert not model.has_edits, "one Ctrl+Z must restore the whole session"

        window.redo()
        assert model.current_value("disabled:buildings:2") == (109, 68)
        assert model.current_value("disabled:techs:5") == (22,)
    finally:
        _close(window)


def test_return_on_a_row_adds_it_and_return_on_a_disabled_row_removes_it() -> None:
    """GH #57: activating a row moves it across. Return fires the same
    itemActivated a double-click does, and is reliable offscreen."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    window = _window()
    try:
        before_depth = len(window.edit_history.records)
        dialog = window._build_disables_dialog()
        dialog.select_player(2)
        full_list = dialog.full_list_for("buildings")
        full_list.select(109)
        assert full_list.current_value() == 109
        QTest.keyClick(full_list.tree, Qt.Key_Return)
        assert dialog.disabled_ids_for("buildings") == (109,)

        disabled_list = dialog._tabs["buildings"].disabled_list
        disabled_list.setCurrentRow(0)
        QTest.keyClick(disabled_list, Qt.Key_Return)
        assert dialog.disabled_ids_for("buildings") == ()
        assert len(window.edit_history.records) == before_depth, "a key reached OK"
        dialog.reject()
    finally:
        _close(window)


@pytest.mark.xfail(strict=True, reason="Return on a list row also presses the default OK button and closes the dialog")
def test_return_on_a_row_of_the_shown_dialog_adds_without_closing_it() -> None:
    """Shown, OK is the default button, and the tree ignores the Return it
    just used, so QDialog accepts: Enter-to-add ends the whole session."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        before_depth = len(window.edit_history.records)
        dialog = window._build_disables_dialog()
        dialog.show()
        QApplication.processEvents()
        full_list = dialog.full_list_for("buildings")
        full_list.select(109)
        QTest.keyClick(full_list.tree, Qt.Key_Return)
        assert dialog.disabled_ids_for("buildings") == (109,)
        assert dialog.isVisible(), "Return on a row closed the dialog"
        assert len(window.edit_history.records) == before_depth
        dialog.reject()
    finally:
        _close(window)


def test_cancel_pushes_nothing_and_leaves_the_document_clean() -> None:
    window = _window()
    try:
        before_depth = len(window.edit_history.records)
        dialog = window._build_disables_dialog()
        dialog.select_player(3)
        dialog.add_id("units", 4)
        dialog.reject()

        assert len(window.edit_history.records) == before_depth
        assert window.option_edits is None or not window.option_edits.has_edits
    finally:
        _close(window)


def test_ok_with_nothing_changed_pushes_nothing() -> None:
    window = _window()
    try:
        before_depth = len(window.edit_history.records)
        window._build_disables_dialog()._accept()
        assert len(window.edit_history.records) == before_depth
    finally:
        _close(window)


def test_switching_player_then_tab_shows_that_players_pending_ids() -> None:
    """The pending copy is per (category, player), so a second dialog opened
    after an OK must show the edited state, not the file's."""
    window = _window()
    try:
        first = window._build_disables_dialog()
        first.select_player(2)
        first.add_id("buildings", 109)
        first._accept()

        second = window._build_disables_dialog()
        second.select_player(2)
        second.select_tab("buildings")
        assert second.disabled_ids_for("buildings") == (109,)
        second.select_player(3)
        assert second.disabled_ids_for("buildings") == (), "P3 should be untouched"
        second.select_player(2)
        assert second.disabled_ids_for("units") == (), "the units tab is its own list"
        assert second.disabled_ids_for("buildings") == (109,)
    finally:
        _close(window)


def test_remove_takes_an_id_back_out() -> None:
    window = _window()
    try:
        dialog = window._build_disables_dialog()
        dialog.select_player(1)
        dialog.add_id("techs", 22)
        dialog.add_id("techs", 23)
        dialog.remove_id("techs", 22)
        assert dialog.disabled_ids_for("techs") == (23,)
        dialog._accept()
        assert window.option_edits.current_value("disabled:techs:1") == (23,)
    finally:
        _close(window)


def test_adding_an_id_twice_is_a_no_op() -> None:
    window = _window()
    try:
        dialog = window._build_disables_dialog()
        dialog.add_id("buildings", 109)
        dialog.add_id("buildings", 109)
        assert dialog.disabled_ids_for("buildings") == (109,)
    finally:
        _close(window)


# -- the ids the Full List cannot produce ------------------------------------


def test_an_out_of_enum_id_renders_with_a_real_name_and_survives_an_untouched_ok(
    tmp_path: Path,
) -> None:
    """621 ("Town Center") is a real corpus value object_catalog.objects()
    does not carry. It has to display as a name and round-trip through a
    dialog session that edited a different list entirely."""
    from descape.options_model import OptionsEditModel
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    # Built through the real write path rather than a hand byte-patch, so
    # the fixture is one the app could actually have produced.
    loaded = load_map_and_units(FIXTURE_PATH)
    seed = OptionsEditModel(loaded)
    seed.set_value("disabled:buildings:1", (621,))
    fixture = tmp_path / "out_of_enum.aoe2scenario"
    write_scenario(loaded, fixture, options=seed)

    window = _window(fixture)
    try:
        dialog = window._build_disables_dialog()
        dialog.select_player(1)
        assert dialog.disabled_ids_for("buildings") == (621,)
        labels = dialog.disabled_labels_for("buildings")
        assert labels and labels[0].endswith("(621)")
        assert not labels[0].startswith("UNKNOWN_"), labels

        dialog.add_id("techs", 22)
        dialog._accept()
        assert window.option_edits.current_value("disabled:buildings:1") == (621,)
        assert window.option_edits.current_value("disabled:techs:1") == (22,)
    finally:
        _close(window)


# -- the Full List panes -----------------------------------------------------


def test_each_tab_offers_only_its_own_category_and_renders_flat() -> None:
    """Exact in-game parity: Buildings offers Buildings, Units offers Units,
    Techs offers the tech enum. Flat, not under one collapsible heading --
    a single-category tree grouped by that category is a click to open and
    nothing to choose between."""
    from descape import object_catalog

    window = _window()
    try:
        dialog = window._build_disables_dialog()
        for category, expected in (("buildings", "Buildings"), ("units", "Units")):
            view = dialog.full_list_for(category)
            ids = {item.value for item in view._items}
            assert ids == {e.id for e in object_catalog.objects() if e.category == expected}
            assert all(item.group == "" for item in view._items)
        techs = dialog.full_list_for("techs")
        assert {item.value for item in techs._items} == {e.id for e in object_catalog.techs()}
    finally:
        _close(window)


def test_the_techs_tab_ships_without_the_dead_hidden_toggle() -> None:
    """object_catalog.json carries no `hidden` flag for a tech, so the box
    would filter nothing. Omitting it is the fix, not a gap -- the objects
    tabs, whose rows do carry the flag, keep theirs."""
    window = _window()
    try:
        dialog = window._build_disables_dialog()
        assert dialog.full_list_for("techs").show_hidden_checkbox is None
        assert dialog.full_list_for("buildings").show_hidden_checkbox is not None
    finally:
        _close(window)


def test_a_carrier_gate_failure_names_the_carrier_not_the_disables_region(monkeypatch) -> None:
    """Two gates, two facts: "this file's map options failed" and "this
    file's disable lists failed" send the reader to different places, so the
    tooltip must name the right one -- the same split the per-player rows
    already make."""
    from descape import viewer as viewer_module
    from descape.players_panel import PlayersPanel

    window = _window()
    try:
        monkeypatch.setattr(viewer_module, "options_write_supported", lambda *_: False)
        window.mode_combo.setCurrentText("Players")
        window._repopulate_players()
        button = window.players_panel.disables_button
        assert not button.isEnabled()
        assert button.toolTip() == PlayersPanel._DISABLES_CARRIER_GATE_REASON
    finally:
        _close(window)
