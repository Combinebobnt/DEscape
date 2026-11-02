"""Swap Players in the window (GH #185): the Players panel's button, Map >
Swap Players…, the dialog, one undo step and the refreshes.

The dialog is built through ViewerWindow._build_swap_dialog(), never shown
with exec_(), which would hang an offscreen run (tests/test_disables_dialog.py
records the same trap). The document is tools/gen_swap_players_probe.py's
before file: P1 and P2 differ on every Players field, in diplomacy, units and
one trigger, and P3 is the witness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

A, B, WITNESS = 1, 2, 3
_close = conftest.close_window


@pytest.fixture(scope="module")
def probe():
    return conftest.load_verify_module("gen_swap_players_probe")


@pytest.fixture(scope="module")
def before_path(probe, tmp_path_factory) -> Path:
    return probe.write_before(probe.SOURCE, tmp_path_factory.mktemp("swap_gui") / "before.aoe2scenario")


@pytest.fixture(scope="module")
def one_player_path(probe, tmp_path_factory) -> Path:
    """The probe's blank saved with Number of Players 1, so Swap starts greyed."""
    from descape import batch_api, player_fields
    from descape.options_model import OptionsEditModel
    from descape.scenario_write import write_scenario

    loaded = batch_api.load(probe.SOURCE)
    options = OptionsEditModel(loaded)
    options.set_value(player_fields.PLAYER_COUNT_FIELD_ID, 1)
    path = tmp_path_factory.mktemp("swap_gui_one") / "one_player.aoe2scenario"
    write_scenario(loaded, path, options=options, backup=False)
    return path


def _window(path: Path):
    window = conftest.shown_window(1200, 800)
    window.load_scenario(path)
    assert window.scenario is not None
    return window


def _values(window, pid: int) -> dict:
    from descape import player_fields

    pending = window._pending_player_values()
    return {
        s.field_id: pending.get(s.field_id, {}).get(pid, player_fields.current_value(window.scenario, s, pid))
        for s in player_fields.specs_for(window.scenario)
        if s.field_id not in ("personality", "player_type")
    }


def _personality(window, pid: int):
    choice = window.option_edits.current_personality(pid)
    return choice.stored_name, choice.ai_type


def _units(window, pid: int) -> list:
    return list(window.scenario.unit_manager.units[pid])


def _stance(window, row: int, col: int) -> int:
    from descape import diplomacy_fields

    cell = diplomacy_fields.stance_cell_id(row, col)
    pending = window._pending_options()
    return pending[cell] if cell in pending else diplomacy_fields.stance_value(window.scenario, row, col)


def _player_attributes(window) -> list:
    from descape.scenario_io import parse_triggers

    return [
        (entry.source_player, entry.target_player)
        for trigger in parse_triggers(window.scenario).triggers
        for entry in (*trigger.conditions, *trigger.effects)
    ]


# -- entry points -------------------------------------------------------------


def test_the_panel_button_and_the_menu_open_the_dialog_with_the_right_first_player(before_path, monkeypatch) -> None:
    from descape import viewer as viewer_module

    opened = []
    monkeypatch.setattr(viewer_module.SwapPlayersDialog, "exec_", lambda self: opened.append(self.choice()) or 0)
    window = _window(before_path)
    try:
        window.mode_combo.setCurrentText("Players")
        panel = window.players_panel
        assert panel.swap_button.isEnabled() and window.swap_players_action.isEnabled()
        assert panel.select_player(B)
        panel.swap_button.click()
        assert opened[-1][:2] == (B, A), "the panel's selected player is A, B the first other"
        assert panel.select_player(0)
        panel.swap_button.click()
        assert opened[-1][0] == A, "GAIA selected opens on P1"
        window.swap_players_action.trigger()
        assert opened[-1][0] == A, "the menu opens on P1"

        dialog = window._build_swap_dialog(WITNESS)
        players = [dialog.player_combo.itemData(i) for i in range(dialog.player_combo.count())]
        partners = [dialog.partner_combo.itemData(i) for i in range(dialog.partner_combo.count())]
        assert players == [A, B, WITNESS] and partners == [A, B], "defined players only, never GAIA or A itself"
        dialog.deleteLater()
    finally:
        _close(window)


def test_the_menu_action_is_registered_rebindable_and_global(tmp_path, before_path, monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    from descape import settings
    from descape import viewer as viewer_module

    assert ("map_swap_players", "Swap Players…", "") in settings.REBINDABLE_ACTIONS
    assert settings.keybind_modes("map_swap_players") == settings.keybind_modes("map_disables") == frozenset()
    (tmp_path / "config.yaml").write_text("keybinds:\n  map_swap_players: F8\n")
    opened = []
    monkeypatch.setattr(viewer_module.SwapPlayersDialog, "exec_", lambda self: opened.append(self) or 0)
    window = _window(before_path)
    try:
        assert window._keybind_actions["map_swap_players"] is window.swap_players_action
        menus = {a.text(): a.menu() for a in window.menuBar().actions() if a.menu() is not None}
        assert window.swap_players_action in menus["&Map"].actions()
        assert window.swap_players_action.shortcut() == QKeySequence("F8")
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        QTest.keyClick(window, Qt.Key_F8)
        QApplication.processEvents()
        assert len(opened) == 1
    finally:
        _close(window)


# -- the dialog -------------------------------------------------------------------


def test_the_preview_line_shows_the_plan_and_a_refusal_disables_swap(probe, before_path, monkeypatch) -> None:
    from descape import options_model
    from descape.player_swap import SwapPlan, SwapRefusal

    window = _window(before_path)
    try:
        dialog = window._build_swap_dialog(A)
        assert dialog.choice()[:2] == (A, B)
        plan = window._plan_swap(A, B, dialog.choice()[2])
        assert isinstance(plan, SwapPlan) and plan.unit_counts == (len(probe.UNITS[A]), len(probe.UNITS[B]))
        assert plan.trigger_writes and plan.diplomacy_cells, "the probe's trigger and grid differ between the two"
        assert dialog.preview_label.text() == plan.summary()
        assert dialog.swap_button.isEnabled()
        assert not window.edit_history.can_undo, "previewing builds no model and records nothing"
        assert (window.option_edits, window.unit_edits, window.trigger_edits) == (None, None, None)

        # The diplomacy gate, where player_swap.file_gates() reads it (the window's own gate builds on it).
        monkeypatch.setattr(options_model, "diplomacy_write_supported", lambda *_: False)
        dialog.diplomacy_check.setChecked(True)
        dialog._update_preview()
        refusal = window._plan_swap(A, B, dialog.choice()[2])
        assert isinstance(refusal, SwapRefusal)
        assert dialog.preview_label.text() == refusal.reason and not dialog.swap_button.isEnabled()
        dialog.diplomacy_check.setChecked(False)
        plan = window._plan_swap(A, B, dialog.choice()[2])
        assert isinstance(plan, SwapPlan), "unticking diplomacy lifts the refusal"
        assert dialog.swap_button.isEnabled() and dialog.preview_label.text() == plan.summary()
        dialog.deleteLater()
    finally:
        _close(window)


def test_the_dialog_lists_swap_partners_as_the_players_a_can_swap_with(before_path, monkeypatch) -> None:
    from descape import player_swap

    real = player_swap.swap_partners
    asked = []

    def partners(loaded, a, pending=None):
        asked.append(a)
        return [p for p in real(loaded, a, pending) if p != B]

    monkeypatch.setattr(player_swap, "swap_partners", partners)
    window = _window(before_path)
    try:
        dialog = window._build_swap_dialog(A)
        listed = [dialog.partner_combo.itemData(i) for i in range(dialog.partner_combo.count())]
        assert asked and asked[-1] == A
        assert listed == [WITNESS], "the partner list is swap_partners()'s, not the dialog's own"
        dialog.deleteLater()
    finally:
        _close(window)


def test_a_longer_refusal_grows_the_dialog_so_the_line_is_not_clipped(before_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape import options_model
    from descape.player_swap import SwapRefusal

    window = _window(before_path)
    try:
        dialog = window._build_swap_dialog(A)
        dialog.show()
        QApplication.processEvents()
        label = dialog.preview_label
        assert label.height() >= label.heightForWidth(label.width())
        monkeypatch.setattr(options_model, "players_write_supported", lambda *_: False)
        monkeypatch.setattr(window.scenario, "units_write_supported", False)
        dialog._update_preview()
        QApplication.processEvents()
        refusal = window._plan_swap(A, B, dialog.choice()[2])
        assert isinstance(refusal, SwapRefusal) and label.text() == refusal.reason
        assert label.heightForWidth(label.width()) > 2 * label.fontMetrics().height()
        assert label.height() >= label.heightForWidth(label.width()), "the refusal is clipped"
        dialog.close()
        dialog.deleteLater()
    finally:
        _close(window)


# -- one undo step ---------------------------------------------------------------


def test_a_swap_is_one_undo_step_that_undo_and_redo_move_whole(before_path) -> None:
    from descape.edit_history import CompositeDiffRecord
    from descape.player_swap import SwapOptions

    window = _window(before_path)
    try:
        window.mode_combo.setCurrentText("Players")
        values = {p: _values(window, p) for p in (A, B, WITNESS)}
        units = {p: _units(window, p) for p in (A, B)}
        stances = (_stance(window, A, B), _stance(window, B, A), _stance(window, A, WITNESS))
        triggers = _player_attributes(window)

        assert window.swap_players(A, B, SwapOptions())
        record = window.edit_history.peek_undo()
        assert isinstance(record, CompositeDiffRecord) and len(window.edit_history.records) == 1
        assert record.kinds() == {"unit", "trigger", "options"}
        assert (_values(window, A), _values(window, B), _values(window, WITNESS)) == (values[B], values[A], values[WITNESS])
        assert (_units(window, A), _units(window, B)) == (units[B], units[A])
        assert (_stance(window, B, A), _stance(window, A, B), _stance(window, B, WITNESS)) == stances
        assert _player_attributes(window) != triggers
        assert window.players_panel.current_values()["tribe_name"] == values[B]["tribe_name"], "the panel shows P1 swapped"
        window.mode_combo.setCurrentText("Diplomacy")
        assert window.diplomacy_panel.select_player(A)
        assert window.diplomacy_panel.current_values()["stance:1:2"] == stances[1]

        window.undo()
        assert not window.option_edits.has_edits
        assert (_values(window, A), _values(window, B)) == (values[A], values[B])
        assert (_units(window, A), _units(window, B)) == (units[A], units[B])
        assert _player_attributes(window) == triggers
        assert window.diplomacy_panel.current_values()["stance:1:2"] == stances[0]

        window.redo()
        assert (_values(window, A), _units(window, A)) == (values[B], units[B])
        assert _stance(window, A, B) == stances[1]
    finally:
        _close(window)


def test_unticking_each_box_leaves_its_category_untouched(before_path) -> None:
    from descape.player_swap import SwapOptions

    window = _window(before_path)
    try:
        triggers = _player_attributes(window)
        stance = _stance(window, A, B)
        assert window.swap_players(A, B, SwapOptions(triggers=False, diplomacy=False))
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
        assert _player_attributes(window) == triggers
        assert _stance(window, A, B) == stance
        assert window.edit_history.peek_undo().kinds() == {"unit", "options"}
    finally:
        _close(window)


def test_a_swap_keeps_the_units_selection_on_the_moved_units(before_path) -> None:
    from descape.player_swap import SwapOptions

    window = _window(before_path)
    try:
        window.mode_combo.setCurrentText("Units")
        unit = _units(window, A)[0]
        window._select_unit_keys([(A, unit.reference_id)])
        assert window._selection == [(A, unit.reference_id)]
        assert window.swap_players(A, B, SwapOptions())
        assert window._selection == [(B, unit.reference_id)], "the inspector keeps its unit"
    finally:
        _close(window)


def test_a_raise_partway_still_pushes_one_composite_that_undo_reverts(before_path, monkeypatch) -> None:
    from descape.player_swap import SwapOptions, SwapPlan

    window = _window(before_path)
    warned = []
    window._warn = lambda title, text: warned.append(title)
    try:
        model = window._ensure_option_edits()
        real = model.set_value
        calls = []

        def flaky(field_id, value):
            calls.append(field_id)
            if len(calls) == 3:
                raise ValueError("injected")
            real(field_id, value)

        monkeypatch.setattr(model, "set_value", flaky)
        assert not window.swap_players(A, B, SwapOptions())
        assert len(warned) == 1 and SwapPlan(A, B, SwapOptions()).label in warned[0], "one warning, naming the swap"
        assert len(window.edit_history.records) == 1 and model.has_edits
        monkeypatch.setattr(model, "set_value", real)
        window.undo()
        assert not model.has_edits
    finally:
        _close(window)


def test_players_that_differ_in_nothing_push_nothing() -> None:
    from descape import player_fields
    from descape.player_swap import SwapOptions, SwapPlan
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = _window(BLANK_TEMPLATE_PATH)
    try:
        color = next(s for s in player_fields.specs_for(window.scenario) if s.field_id == "color")
        window.set_player_field(color, B, player_fields.current_value(window.scenario, color, A))
        plan = window._plan_swap(A, B, SwapOptions())
        assert isinstance(plan, SwapPlan) and plan.is_empty
        count = len(window.edit_history.records)
        assert not window.swap_players(A, B, SwapOptions())
        assert len(window.edit_history.records) == count
    finally:
        _close(window)


def test_a_data_locked_document_refuses(before_path, monkeypatch) -> None:
    from descape.player_swap import SwapOptions

    window = _window(before_path)
    try:
        monkeypatch.setattr(window, "game_data_lock_reason", lambda: "Shown with the other game's data.")
        window._update_tool_enabled()
        window.mode_combo.setCurrentText("Players")
        window._repopulate_players()
        assert not window.swap_players_action.isEnabled()
        assert not window.players_panel.swap_button.isEnabled()
        assert window._build_swap_dialog(A) is None
        assert not window.swap_players(A, B, SwapOptions())
        assert not window.edit_history.records and window.option_edits is None
    finally:
        _close(window)


def test_the_swap_gates_are_the_file_gates_closed_whole_by_the_data_lock(before_path, monkeypatch) -> None:
    from descape import options_model, player_swap

    window = _window(before_path)
    try:
        assert window._swap_gates() == player_swap.file_gates(window.scenario)
        monkeypatch.setattr(options_model, "diplomacy_write_supported", lambda *_: False)
        assert not window._swap_gates().diplomacy, "read where file_gates() reads it, not a second copy"
        monkeypatch.setattr(window, "game_data_lock_reason", lambda: "Shown with the other game's data.")
        assert window._swap_gates() == player_swap.SwapGates(), "GH #67 closes every gate, units included"
    finally:
        _close(window)


def test_the_swap_gates_skip_the_custom_ai_check_when_personality_is_not_writable(before_path, monkeypatch) -> None:
    from descape import options_model

    window = _window(before_path)
    try:
        calls = []
        monkeypatch.setattr(window, "_personality_custom_ok", lambda: calls.append(1) or True)
        monkeypatch.setattr(options_model, "personality_write_supported", lambda *_: False)
        assert not window._swap_gates().personality_custom
        assert not calls, "the check forces the cold trigger parse"
    finally:
        _close(window)


def test_a_player_count_edit_and_its_undo_and_redo_regate_both_entry_points(one_player_path) -> None:
    window = _window(one_player_path)
    try:
        window.mode_combo.setCurrentText("Players")
        panel = window.players_panel

        def gates() -> tuple[bool, bool]:
            return window.swap_players_action.isEnabled(), panel.swap_button.isEnabled()

        assert window._current_player_count() == 1 and gates() == (False, False)
        panel.player_count_spin.setValue(2)
        assert gates() == (True, True), "1 -> 2 players"
        panel.player_count_spin.setValue(1)
        assert gates() == (False, False), "2 -> 1 players"
        window.undo()
        assert gates() == (True, True), "undo back to 2"
        window.undo()
        assert gates() == (False, False), "undo back to 1"
        window.redo()
        assert gates() == (True, True), "redo to 2"
        window.redo()
        assert gates() == (False, False), "redo to 1"
        window.mode_combo.setCurrentText("Diplomacy")
        window.undo()
        assert gates() == (True, True), "an undo outside Players mode re-gates both too"
    finally:
        _close(window)


@pytest.mark.font_sensitive
def test_the_panel_buttons_fit_min_useful_width(before_path) -> None:
    from descape.players_panel import PlayersPanel

    window = _window(before_path)
    try:
        row = window.players_panel.scenario_buttons
        assert window.players_panel.swap_button.parentWidget() is row
        width = row.minimumSizeHint().width()
        assert 0 < width <= PlayersPanel.MIN_USEFUL_WIDTH, f"the button row demands {width} px"
    finally:
        _close(window)


@pytest.mark.test_map
def test_a_swap_of_the_test_maps_two_biggest_players_completes_as_one_step() -> None:
    """The Mirror Map-sized refresh (a recolour and two big lists) finishes; no new perf claim."""
    from descape.player_swap import SwapOptions

    window = _window(conftest.TEST_MAP_PATH)
    try:
        sizes = (len(_units(window, A)), len(_units(window, B)))
        assert window.swap_players(A, B, SwapOptions())
        assert (len(_units(window, A)), len(_units(window, B))) == sizes[::-1]
        window.undo()
        assert (len(_units(window, A)), len(_units(window, B))) == sizes
    finally:
        _close(window)
