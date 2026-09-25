"""GH #5: View mode's per-player stats combo and block, driven through a real
offscreen ViewerWindow. The counts themselves are tests/test_player_stats.py."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNITS_FIXTURE = FIXTURES / "units_120x120.aoe2scenario"
TRIGGER_FIXTURE = FIXTURES / "triggers_120x120.aoe2scenario"
_PLACE_CONST = 83


def _window(path: Path):
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(path)
    assert window.scenario is not None, f"{path.name} failed to load"
    return window


def _row(window, label: str) -> str:
    """The row's count and note as displayed, checked against the label's own text."""
    rows = {row[0]: row for row in window.player_stats_rows}
    assert label in rows, window.player_stats_rows
    _, count, note = rows[label]
    text = window.player_stats_label.text()
    assert f"<td>{label}</td>" in text
    assert all(part in text for part in (count, note.replace("&", "&amp;")) if part)
    return " ".join(part for part in (count, note) if part)


def _placements(window) -> int:
    return int(_row(window, "Placements").replace(",", ""))


def _combo_items(window) -> list[tuple[str, int]]:
    combo = window.stats_player_combo
    return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]


def test_the_combo_lists_gaia_then_the_defined_players_and_defaults_to_player_one() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        assert _combo_items(window) == [("GAIA", 0), ("Player 1", 1), ("Player 2", 2)]
        assert window.stats_player_combo.currentData() == 1
        assert window.player_stats_label.text().startswith("<b>Player 1</b>")
        assert _placements(window) == 3
        assert "Units per player" not in window.info.toPlainText()
    finally:
        conftest.close_window(window)


def test_every_combo_item_carries_its_players_colour_swatch() -> None:
    """GH #5 step 2: GAIA included, each item's icon is its player's colour."""
    window = _window(UNITS_FIXTURE)
    try:
        combo = window.stats_player_combo
        assert combo.count() == 3
        for i in range(combo.count()):
            icon = combo.itemIcon(i)
            assert not icon.isNull(), combo.itemText(i)
            pixel = icon.pixmap(16, 16).toImage().pixelColor(8, 8)
            expected = tuple(window.scenario.player_colors[combo.itemData(i)])
            assert (pixel.red(), pixel.green(), pixel.blue()) == expected, combo.itemText(i)
    finally:
        conftest.close_window(window)


def test_lowering_the_player_count_labels_a_slot_that_owns_units_inactive() -> None:
    """GH #5 step 7 through a real Number of Players edit. The fixture sits at
    the spinbox's minimum of 2, so it first gains a P3 unit and a third slot."""
    window = _window(UNITS_FIXTURE)
    try:
        model = window._ensure_unit_edits()
        assert model is not None
        with window._unit_edit(model, "Place for P3", [3]):
            model.add(3, _PLACE_CONST, 30.5, 30.5)
        window.mode_combo.setCurrentText("Players")
        spin = window.players_panel.player_count_spin
        assert spin.isEnabled(), "Number of Players is read-only on this fixture"
        assert spin.value() == 2
        assert _combo_items(window)[-1] == ("Player 3 (inactive)", 3)

        spin.setValue(3)
        assert _combo_items(window)[-1] == ("Player 3", 3)
        spin.setValue(2)
        assert _combo_items(window)[-1] == ("Player 3 (inactive)", 3)

        window.undo()
        assert spin.value() == 3
        assert _combo_items(window)[-1] == ("Player 3", 3)
    finally:
        conftest.close_window(window)


def test_an_inactive_slot_that_owns_placements_is_listed_and_labelled(monkeypatch) -> None:
    window = _window(UNITS_FIXTURE)
    try:
        monkeypatch.setattr(window, "_current_active_players", lambda: [1])
        window._repopulate_stats_players()
        assert _combo_items(window) == [("GAIA", 0), ("Player 1", 1), ("Player 2 (inactive)", 2)]
    finally:
        conftest.close_window(window)


def test_player_select_keybinds_drive_the_combo_in_view_mode() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        assert window.mode == "view"
        window._select_player(0)
        assert window.stats_player_combo.currentData() == 0
        assert window.player_stats_label.text().startswith("<b>GAIA</b>")
        window._select_player(2)
        assert window.stats_player_combo.currentData() == 2
        assert _placements(window) == 2
        window._select_player(5)  # not defined: selection stays put
        assert window.stats_player_combo.currentData() == 2
    finally:
        conftest.close_window(window)


def test_placing_and_undoing_a_unit_moves_the_count_both_ways() -> None:
    """The regression guard for the old per-player list, which only
    refreshed on load, Save As and a trigger-parse mode switch."""
    from PyQt5.QtCore import QPointF, Qt

    window = _window(UNITS_FIXTURE)
    try:
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        window.mode_combo.setCurrentText("Units")
        assert window.stats_player_combo.currentData() == 1
        before = _placements(window)
        window.units_panel.select_object(_PLACE_CONST)
        window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        tp = window.map_view._tile_pixels
        window.on_unit_place(QPointF(10 * tp + tp // 2, 10 * tp + tp // 2), Qt.NoModifier)
        assert _placements(window) == before + 1
        window.undo()
        assert _placements(window) == before
        window.redo()
        assert _placements(window) == before + 1
    finally:
        conftest.close_window(window)


def test_view_mode_never_pays_the_trigger_parse() -> None:
    window = _window(TRIGGER_FIXTURE)
    try:
        assert window.mode == "view"
        window._select_player(0)
        window._select_player(1)
        assert window.scenario.trigger_read_supported is None
        assert _row(window, "Triggers") == "not counted yet (enter Triggers mode or run Map Analysis)"
    finally:
        conftest.close_window(window)


def test_map_analysis_updates_both_info_halves() -> None:
    """_show_analysis() parses the triggers but used to leave the info panel
    saying they weren't, since the mode-switch refresh never saw the flip."""
    window = _window(TRIGGER_FIXTURE)
    try:
        assert "Embedded XS: (unknown until triggers are parsed)" in window.info.toPlainText()
        window._show_analysis()
        assert window.scenario.trigger_read_supported is True
        assert "Embedded XS: (none)" in window.info.toPlainText()
        assert re.fullmatch(r"2 +\(of 4; 2 reference no player\)", _row(window, "Triggers"))
    finally:
        window._close_analysis_dialog()
        conftest.close_window(window)


def test_entering_triggers_mode_turns_the_hint_into_a_count() -> None:
    window = _window(TRIGGER_FIXTURE)
    try:
        window.mode_combo.setCurrentText("Triggers")
        window.mode_combo.setCurrentText("View")
        assert re.fullmatch(r"2 +\(of 4; 2 reference no player\)", _row(window, "Triggers"))
    finally:
        conftest.close_window(window)


def test_an_unparseable_triggers_section_reads_unavailable_not_zero() -> None:
    from descape import map_analysis

    window = _window(TRIGGER_FIXTURE)
    try:
        window.scenario.trigger_read_supported = False
        window._update_player_stats()
        assert _row(window, "Triggers") == map_analysis._TRIGGERS_UNAVAILABLE
    finally:
        conftest.close_window(window)


def test_closing_the_scenario_clears_the_combo() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        window.close_scenario()
        assert window.stats_player_combo.count() == 0
        assert not window.stats_player_combo.isEnabled()
        assert window.player_stats_label.text() == ""
    finally:
        conftest.close_window(window)
