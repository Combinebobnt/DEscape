"""GH #169: the button grid under button_location on Change Train Location,
Add Train Location and Change Research Location.

Same offscreen-window technique as tests/test_trigger_panel.py, whose
helpers build the effects. No install, so unit cells show elided names.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_CHANGE_TRAIN, _CHANGE_RESEARCH, _ADD_TRAIN = 46, 47, 102
_BARRACKS, _DOCK, _STABLE, _BUILDER, _VILLAGER, _ARMY_TENT = 12, 45, 101, 118, 83, 1097
_MILITIA, _SPEARMAN, _SCOUT, _KNIGHT, _MARKET, _CASTLE, _LOOM = 74, 93, 448, 38, 84, 82, 22
_PIKEMAN = 358


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _grid_window(type_id: int, *per_entry):
    from test_trigger_panel import _facet_window

    return _facet_window(type_id, *per_entry)


def _grid(window):
    grid = window.trigger_panel._button_grid
    assert grid is not None, "no button grid on this form"
    return grid


def _cell(grid, row: int, col: int):
    return grid.cells[(row - 1) * 5 + col - 1]


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def test_a_lower_id_replaces_the_slots_button() -> None:
    """In game, Knight 38 at Barracks button 2 hid the Spearman line (93)."""
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _KNIGHT, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        grid = _grid(window)
        assert not grid.isHidden()
        assert grid.header_label.text() == "Trained at: Barracks (any civ: player 1's civ is random)"
        chosen = _cell(grid, 1, 2)
        assert chosen.isChecked() and [c for c in grid.cells if c.isChecked()] == [chosen]
        assert chosen.items == ((_KNIGHT, "unit"),) and not chosen.blocked
        assert chosen.replaced[0] == (_SPEARMAN, "unit")
        assert "Spearman" not in chosen.toolTip(), "only the new button shows"
        assert grid.footer_label.text() == "Now at: Stable, row 1, col 2", "no button slot line (user)"
        assert _cell(grid, 1, 1).items[0] == (_MILITIA, "unit")
        assert grid.page == "page 1" and grid.page_buttons["page 2"].isVisible(), "every building has page 2"
    finally:
        _close(window)


def test_a_higher_id_is_hidden_by_the_slots_button() -> None:
    """In game, Scout Cavalry 448 at Barracks button 2 did not show over the Spearman line."""
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        grid = _grid(window)
        chosen = _cell(grid, 1, 2)
        assert chosen.isChecked() and chosen.blocked
        assert chosen.items[0] == (_SPEARMAN, "unit")
        assert chosen.toolTip().endswith(
            "Hidden here: Scout Cavalry (the lower id shows). Move or disable Spearman to free the slot."
        )
        assert _cell(grid, 1, 1).items[0] == (_MILITIA, "unit")
    finally:
        _close(window)


def test_a_unit_already_in_its_slot_still_lists_its_slot_mates() -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _SPEARMAN, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        chosen = _cell(_grid(window), 1, 2)
        assert chosen.isChecked() and chosen.items[0] == (_SPEARMAN, "unit")
        assert (_PIKEMAN, "unit") in chosen.items
        assert "Pikeman" in chosen.toolTip()
    finally:
        _close(window)


def test_moving_away_a_unit_that_is_not_the_cells_icon_leaves_the_icon_undimmed() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap

    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _PIKEMAN, "object_list_unit_id_2": _BARRACKS, "button_location": 3}
    )
    try:
        grid = _grid(window)
        cell = _cell(grid, 1, 2)
        assert cell.dimmed and cell.items[0] == (_SPEARMAN, "unit")
        solid = QPixmap(8, 8)
        solid.fill(Qt.red)
        grid._icon = lambda _id, _kind: solid
        assert grid._cell_icon(cell).pixmap(8, 8).toImage().pixelColor(4, 4).alpha() == 255
    finally:
        _close(window)


def test_the_tooltip_lists_every_occupant() -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _CASTLE, "button_location": 5}
    )
    try:
        cell = _cell(_grid(window), 1, 1)
        assert len(cell.items) > 50
        assert cell.toolTip().count("\n") == 12, "12 names and a +N more line"
        assert cell.toolTip().endswith(f"+{len(cell.items) - 12} more")
    finally:
        _close(window)


def test_the_grid_follows_unit_building_and_slot_edits() -> None:
    from PyQt5.QtWidgets import QApplication
    from test_trigger_panel import _row_widget

    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        panel = window.trigger_panel
        building = _row_widget(panel, "object_list_unit_id_2")
        building.set_value(_STABLE)
        building.committed.emit(_STABLE)
        QApplication.processEvents()
        grid = _grid(window)
        assert grid.header_label.text() == "Trained at: Stable (any civ: player 1's civ is random)"
        assert _cell(grid, 1, 1).dimmed, "the scout's old Stable button moved away"
        assert "Moved away: Scout Cavalry" in _cell(grid, 1, 1).toolTip()

        unit = _row_widget(panel, "object_list_unit_id")
        unit.set_value(_MILITIA)
        unit.committed.emit(_MILITIA)
        QApplication.processEvents()
        grid = _grid(window)
        assert _cell(grid, 1, 2).blocked, "the knight (38) keeps Stable button 2"
        assert not any(cell.dimmed for cell in grid.cells)

        combo = _row_widget(panel, "button_location")
        combo.setCurrentIndex(combo.findData(9))
        QApplication.processEvents()
        assert _cell(_grid(window), 2, 4).isChecked()
    finally:
        _close(window)


def test_add_train_location_keeps_the_old_button() -> None:
    window, _indices = _grid_window(
        _ADD_TRAIN, {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _STABLE, "button_location": 5}
    )
    try:
        grid = _grid(window)
        assert not any(cell.dimmed for cell in grid.cells)
        assert (_SCOUT, "unit") in _cell(grid, 1, 1).items
    finally:
        _close(window)


def test_a_cell_click_writes_button_location_as_one_undo_step() -> None:
    from PyQt5.QtWidgets import QApplication

    window, indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        effect = window.trigger_panel._manager().triggers[0].effects[indices[0]]
        before = len(window.edit_history.records)
        _cell(_grid(window), 2, 2).click()
        QApplication.processEvents()
        assert effect.button_location == 7
        assert len(window.edit_history.records) == before + 1
        assert window.edit_history.records[-1].label == "Set effect button slot"
        assert _cell(_grid(window), 2, 2).isChecked()

        # The chosen cell again writes nothing and stays checked.
        _cell(_grid(window), 2, 2).click()
        QApplication.processEvents()
        assert len(window.edit_history.records) == before + 1
        assert _cell(_grid(window), 2, 2).isChecked()

        window.undo()
        QApplication.processEvents()
        assert window.trigger_edits.manager().triggers[0].effects[indices[0]].button_location == 2
    finally:
        _close(window)


def test_the_dock_has_a_page_switch_and_opens_on_the_values_page() -> None:
    from PyQt5.QtWidgets import QApplication

    window, indices = _grid_window(
        _CHANGE_RESEARCH, {"technology": _LOOM, "object_list_unit_id_2": _DOCK, "button_location": 27}
    )
    try:
        grid = _grid(window)
        assert grid.header_label.text() == "Researched at: Dock (any civ: player 1's civ is random)"
        assert grid.page == "page 2"
        assert grid.page_buttons["page 1"].isVisible() and grid.page_buttons["page 2"].isChecked()
        chosen = _cell(grid, 2, 2)
        assert chosen.isChecked() and chosen.items[0] == (_LOOM, "tech") and chosen.shared, "shares with Dragon Ship"
        assert chosen.text(), "a tech has no icon, so its name"
        assert "Now at: Town Center, row 2, col 1" in grid.footer_label.text()

        grid.page_buttons["page 1"].click()
        assert grid.page == "page 1" and not any(cell.isChecked() for cell in grid.cells)
        _cell(grid, 3, 5).click()
        QApplication.processEvents()
        assert window.trigger_panel._manager().triggers[0].effects[indices[0]].button_location == 15
    finally:
        _close(window)


def test_a_tech_takes_turns_with_the_slots_techs() -> None:
    """In game, Bloodlines at Barracks button 12 and Squires showed one after the other."""
    window, _indices = _grid_window(
        _CHANGE_RESEARCH, {"technology": 435, "object_list_unit_id_2": _BARRACKS, "button_location": 12}
    )
    try:
        cell = _cell(_grid(window), 3, 2)
        assert cell.isChecked() and cell.shared and not cell.replaced
        assert cell.items[0] == (435, "tech") and (215, "tech") in cell.items
        assert cell.toolTip().endswith("the first one researchable shows, then the next.")
        assert "Squires" in cell.toolTip()
    finally:
        _close(window)


def test_the_builder_has_economic_and_military_menus() -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _CASTLE, "object_list_unit_id_2": _BUILDER, "button_location": 12}
    )
    try:
        grid = _grid(window)
        assert grid.header_label.text() == "Trained at: Villager build menu (any civ: player 1's civ is random)"
        assert grid.page == "military", "opens on the castle's own menu"
        assert grid.page_buttons["economic"].isVisible() and grid.page_buttons["military"].isVisible()
        assert _cell(grid, 3, 2).isChecked() and _cell(grid, 3, 2).items == ((_CASTLE, "unit"),)
        assert [_cell(grid, 3, col).reserved for col in (4, 5)] == ["More", "Cancel"]
        assert _cell(grid, 3, 3).dimmed, "the castle's old military button"
        assert "Now at: Villager build menu (military), row 3, col 3" in grid.footer_label.text()

        grid.page_buttons["economic"].click()
        assert grid.page == "economic"
        assert _cell(grid, 1, 1).items[0] == (70, "unit"), "House"
        assert not any(cell.isEnabled() for cell in grid.cells), "the other menu is for looking"
    finally:
        _close(window)


def test_a_builder_menus_own_buttons_hide_a_building() -> None:
    """In game, Castle at builder button 15 had no button, though its hotkey worked."""
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _CASTLE, "object_list_unit_id_2": _BUILDER, "button_location": 15}
    )
    try:
        grid = _grid(window)
        cell = _cell(grid, 3, 5)
        assert cell.isChecked() and cell.reserved == "Cancel" and cell.text() == "Cancel"
        assert cell.items == () and not cell.isEnabled()
        assert "hotkey still works" in cell.toolTip()
        _cell(grid, 1, 1).click()
        assert not _cell(_grid(window), 1, 1).reserved, "other builders' slots stay plain"
    finally:
        _close(window)


def _combo_values(combo) -> list:
    return [combo.itemData(i) for i in range(combo.count())]


def test_the_builder_offers_slots_1_to_13_and_follows_the_building() -> None:
    from PyQt5.QtWidgets import QApplication
    from test_trigger_panel import _row_widget

    window, indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _CASTLE, "object_list_unit_id_2": _BUILDER, "button_location": 12}
    )
    try:
        panel = window.trigger_panel
        combo = _row_widget(panel, "button_location")
        assert _combo_values(combo) == list(range(14)), "no More, Cancel or page 2"
        assert combo.currentData() == 12

        building = _row_widget(panel, "object_list_unit_id_2")
        building.set_value(_BARRACKS)
        building.committed.emit(_BARRACKS)
        QApplication.processEvents()
        assert _combo_values(combo) == [0, *range(1, 16), *range(21, 36)]
        assert combo.currentData() == 12 and combo.itemText(combo.currentIndex()) == "row 3, col 2 (12)"

        building.set_value(_BUILDER)
        building.committed.emit(_BUILDER)
        QApplication.processEvents()
        assert _combo_values(combo) == list(range(14))
        effect = panel._manager().triggers[0].effects[indices[0]]
        assert effect.button_location == 12, "trimming the rows writes nothing"
    finally:
        _close(window)


def test_a_builder_value_outside_1_to_13_keeps_its_row() -> None:
    from test_trigger_panel import _row_widget

    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _CASTLE, "object_list_unit_id_2": _BUILDER, "button_location": 15}
    )
    try:
        combo = _row_widget(window.trigger_panel, "button_location")
        assert _combo_values(combo) == [*range(14), 15]
        assert combo.currentData() == 15
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_a_villager_as_trained_at_points_to_the_builder() -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _CASTLE, "object_list_unit_id_2": _VILLAGER, "button_location": 14}
    )
    try:
        grid = _grid(window)
        assert grid.grid_host.isHidden()
        assert "builder (118)" in grid.hint_label.text()
    finally:
        _close(window)


def test_unset_or_buttonless_shows_a_hint_instead_of_the_grid() -> None:
    from test_trigger_panel import _row_widget

    window, _indices = _grid_window(_CHANGE_TRAIN, {"object_list_unit_id_2": _BARRACKS, "button_location": 1})
    try:
        grid = _grid(window)
        assert not grid.isHidden() and grid.grid_host.isHidden()
        assert grid.hint_label.text() == "Pick the unit to train to preview the building's buttons."

        unit = _row_widget(window.trigger_panel, "object_list_unit_id")
        unit.set_value(_SCOUT)
        unit.committed.emit(_SCOUT)
        building = _row_widget(window.trigger_panel, "object_list_unit_id_2")
        building.set_value(_MILITIA)
        building.committed.emit(_MILITIA)
        grid = _grid(window)
        assert grid.grid_host.isHidden()
        assert grid.hint_label.text().endswith("is not a building and has no command buttons in the game data.")
    finally:
        _close(window)


def test_a_building_with_no_buttons_shows_an_empty_panel() -> None:
    """In game, Battle Elephants trained at Army Tent A showed on both pages."""
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": 1132, "object_list_unit_id_2": _ARMY_TENT, "button_location": 22}
    )
    try:
        grid = _grid(window)
        assert not grid.grid_host.isHidden() and grid.hint_label.isHidden()
        assert grid.page == "page 2" and grid.page_buttons["page 1"].isVisible()
        assert _cell(grid, 1, 2).isChecked() and _cell(grid, 1, 2).items == ((1132, "unit"),)
        assert all(not cell.items for cell in grid.cells if cell is not _cell(grid, 1, 2))
        grid.page_buttons["page 1"].click()
        assert all(not cell.items for cell in grid.cells)
    finally:
        _close(window)


def test_a_read_only_file_disables_the_cells() -> None:
    window, indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        panel = window.trigger_panel
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        panel.select_trigger(0)
        panel.refresh_entries(select=("effect", indices[0]))
        grid = _grid(window)
        assert not any(cell.isEnabled() for cell in grid.cells)
        _cell(grid, 2, 2).click()
        records = len(window.edit_history.records)
        grid._cell_clicked(_cell(grid, 2, 2))
        assert panel._manager().triggers[0].effects[indices[0]].button_location == 2
        assert len(window.edit_history.records) == records
    finally:
        _close(window)


def test_a_multi_selection_shows_no_grid() -> None:
    from test_trigger_panel import _select_entries

    window, indices = _grid_window(
        _CHANGE_TRAIN,
        {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _BARRACKS, "button_location": 2},
        {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _BARRACKS, "button_location": 2},
    )
    try:
        _select_entries(window.trigger_panel, ("effect", indices[0]), ("effect", indices[1]))
        assert window.trigger_panel._button_grid is None
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_the_grid_takes_no_focus() -> None:
    """A focusable grid child would pull focus off a line edit and fire its commit."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QWidget

    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        grid = _grid(window)
        for widget in [grid, *grid.findChildren(QWidget)]:
            assert widget.focusPolicy() == Qt.NoFocus, widget
    finally:
        _close(window)


# -- GH #230: civ filter, green frame, covered techs, all triggers' changes --------

_CONQUISTADOR, _LONGBOWMAN, _MULE_CART = 771, 8, 1808
_SPANISH, _BRITONS = 14, 1
_TOWN_CENTER = 109


@pytest.fixture
def civs(monkeypatch):
    """Player 1 Spanish, player 2 Britons, every other player random."""
    from descape import civ_availability

    real = civ_availability.player_civ_value
    by_player = {1: _SPANISH, 2: _BRITONS}
    monkeypatch.setattr(
        civ_availability, "player_civ_value", lambda loaded, player: by_player.get(player) or real(loaded, player)
    )


def _ids(cell) -> set[int]:
    return {id_ for id_, _kind in cell.items}


def test_the_grid_shows_only_what_the_players_civ_has(civs) -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN,
        {"object_list_unit_id": _KNIGHT, "object_list_unit_id_2": _CASTLE, "button_location": 5, "source_player": 1},
    )
    try:
        grid = _grid(window)
        assert grid.header_label.text() == "Trained at: Castle (Spanish)"
        unique = _ids(_cell(grid, 1, 1))
        assert _CONQUISTADOR in unique and _LONGBOWMAN not in unique
        assert unique <= {_CONQUISTADOR, 773}, "only the Spanish unique unit and its elite"
    finally:
        _close(window)


def test_the_civ_filter_follows_source_player(civs) -> None:
    from PyQt5.QtWidgets import QApplication
    from test_trigger_panel import _row_widget

    window, _indices = _grid_window(
        _CHANGE_TRAIN,
        {"object_list_unit_id": _KNIGHT, "object_list_unit_id_2": _CASTLE, "button_location": 5, "source_player": 1},
    )
    try:
        combo = _row_widget(window.trigger_panel, "source_player")
        combo.setCurrentIndex(combo.findData(2))
        QApplication.processEvents()
        grid = _grid(window)
        assert grid.context.civ == _BRITONS
        assert _LONGBOWMAN in _ids(_cell(grid, 1, 1)) and _CONQUISTADOR not in _ids(_cell(grid, 1, 1))

        combo.setCurrentIndex(combo.findData(3))
        QApplication.processEvents()
        grid = _grid(window)
        assert grid.header_label.text() == "Trained at: Castle (any civ: player 3's civ is random)"
        assert {_LONGBOWMAN, _CONQUISTADOR} <= _ids(_cell(grid, 1, 1)), "a random civ filters nothing"
    finally:
        _close(window)


def test_a_unit_the_civ_lacks_gets_no_button_and_a_red_frame(civs) -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN,
        {"object_list_unit_id": _MULE_CART, "object_list_unit_id_2": _BARRACKS, "button_location": 8, "source_player": 1},
    )
    try:
        cell = _cell(_grid(window), 2, 3)
        assert cell.isChecked() and cell.unavailable and cell.frame == "problem"
        assert cell.items == () and cell.ghost == (_MULE_CART, "unit"), "drawn dimmed so the slot shows what is missing"
        # The rasteriser picks the cut: "Mu…" under Linux's FreeType, "M…" on Windows (TASK-409).
        text = cell.text()
        assert len(text) > 1 and "Mule Cart".startswith(text.removesuffix("…")), "no install: its elided name"
        assert "Spanish can't train Mule Cart" in cell.toolTip()
    finally:
        _close(window)


def test_the_effects_own_button_gets_a_green_frame() -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _KNIGHT, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        grid = _grid(window)
        assert [cell for cell in grid.cells if cell.frame] == [_cell(grid, 1, 2)]
        assert _cell(grid, 1, 2).frame == "ok"
        grid.grab()  # paints the frame without error
    finally:
        _close(window)


def test_a_hidden_button_gets_a_red_frame() -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        cell = _cell(_grid(window), 1, 2)
        assert cell.blocked and cell.frame == "problem"
    finally:
        _close(window)


def test_a_unit_on_a_techs_slot_says_it_covers_the_tech() -> None:
    """In game, a Mule Cart at Barracks button 6 left Long Swordsman unresearchable."""
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _KNIGHT, "object_list_unit_id_2": _BARRACKS, "button_location": 6}
    )
    try:
        cell = _cell(_grid(window), 2, 1)
        assert cell.items == ((_KNIGHT, "unit"),) and cell.frame == "problem", "red: it costs a tech"
        assert (222, "tech") in cell.replaced
        assert "can't be researched while this unit is here" in cell.toolTip()
    finally:
        _close(window)


def test_a_tech_on_a_units_slot_is_covered() -> None:
    window, _indices = _grid_window(
        _CHANGE_RESEARCH, {"technology": _LOOM, "object_list_unit_id_2": _BARRACKS, "button_location": 1}
    )
    try:
        cell = _cell(_grid(window), 1, 1)
        assert cell.blocked and cell.frame == "problem"
        assert cell.items[0] == (_MILITIA, "unit")
        assert "a unit in the slot covers it" in cell.toolTip()
    finally:
        _close(window)


def test_show_all_triggers_applies_the_players_other_effects() -> None:
    from descape import settings

    window, _indices = _grid_window(
        _CHANGE_TRAIN,
        {"object_list_unit_id": _SCOUT, "object_list_unit_id_2": _BARRACKS, "button_location": 8, "source_player": 1},
        {"object_list_unit_id": _KNIGHT, "object_list_unit_id_2": _BARRACKS, "button_location": 10, "source_player": 1},
        {"object_list_unit_id": _SPEARMAN, "object_list_unit_id_2": _BARRACKS, "button_location": 14, "source_player": 2},
    )
    try:
        grid = _grid(window)
        assert not grid.all_triggers and not grid.all_triggers_check.isHidden()
        assert _KNIGHT not in _ids(_cell(grid, 2, 5))

        grid.all_triggers_check.setChecked(True)
        assert settings.get_button_grid_all_triggers() is True
        knight = _cell(grid, 2, 5)
        assert knight.items[0] == (_KNIGHT, "unit") and knight.frame == "ok"
        assert ": Change Train Location: " in knight.toolTip() and knight.toolTip().startswith("Knight")
        assert _cell(grid, 2, 3).frame == "ok", "the effect's own button keeps its frame"
        assert _SPEARMAN in _ids(_cell(grid, 1, 2)) and not _cell(grid, 3, 4).frame, "player 2's effect is left out"

        grid.all_triggers_check.setChecked(False)
        assert settings.get_button_grid_all_triggers() is False
        assert not _cell(grid, 2, 5).frame
    finally:
        _close(window)


def test_a_tech_cell_uses_the_installs_tech_icon(monkeypatch) -> None:
    import numpy as np

    from descape import asset_source

    seen = []

    def icon(index):
        seen.append(index)
        return np.full((8, 8, 4), 200, dtype=np.uint8)

    monkeypatch.setattr(asset_source, "get_tech_icon", icon)
    window, _indices = _grid_window(
        _CHANGE_RESEARCH, {"technology": _LOOM, "object_list_unit_id_2": _TOWN_CENTER, "button_location": 6}
    )
    try:
        cell = _cell(_grid(window), 2, 1)
        assert cell.items == ((_LOOM, "tech"),)
        assert not cell.icon().isNull() and cell.text() == ""
        assert 6 in seen, "Loom's catalog icon is 6 (006_loom.DDS)"
    finally:
        _close(window)


def test_another_effects_hidden_button_gets_a_red_frame() -> None:
    """A tech put on Barracks button 1 by another effect stays under the Militia (measured with Chemistry)."""
    window, _indices = _grid_window(
        _CHANGE_TRAIN,
        {"object_list_unit_id": _KNIGHT, "object_list_unit_id_2": _BARRACKS, "button_location": 8, "source_player": 1},
    )
    try:
        from test_trigger_panel import _appended_effects

        _appended_effects(
            window, _CHANGE_RESEARCH, [{"technology": _LOOM, "object_list_unit_id_2": _BARRACKS, "button_location": 1, "source_player": 1}]
        )
        window.trigger_panel.refresh_entries(select=("effect", _indices[0]))
        grid = _grid(window)
        grid.all_triggers_check.setChecked(True)
        cell = _cell(grid, 1, 1)
        assert cell.items[0] == (_MILITIA, "unit") and cell.frame == "problem"
        assert "Loom here (hidden)" in cell.toolTip()
        assert grid._placed_problem(((_KNIGHT, "unit"), (222, "tech")), (_KNIGHT, "unit")) == " (covers a tech)"
    finally:
        _close(window)


def test_another_effects_unit_the_civ_lacks_gets_a_red_ghost(civs) -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN,
        {"object_list_unit_id": _KNIGHT, "object_list_unit_id_2": _BARRACKS, "button_location": 10, "source_player": 1},
        {"object_list_unit_id": _MULE_CART, "object_list_unit_id_2": _BARRACKS, "button_location": 8, "source_player": 1},
    )
    try:
        grid = _grid(window)
        assert _cell(grid, 2, 3).ghost is None, "only with the toggle"
        grid.all_triggers_check.setChecked(True)
        cell = _cell(grid, 2, 3)
        assert cell.items == () and cell.ghost == (_MULE_CART, "unit") and cell.frame == "problem"
        assert "Mule Cart here, but Spanish can't train Mule Cart" in cell.toolTip()
    finally:
        _close(window)


def test_the_toggle_sits_below_the_grid() -> None:
    window, _indices = _grid_window(
        _CHANGE_TRAIN, {"object_list_unit_id": _KNIGHT, "object_list_unit_id_2": _BARRACKS, "button_location": 2}
    )
    try:
        grid = _grid(window)
        layout = grid.layout()
        assert layout.indexOf(grid.all_triggers_check) > layout.indexOf(grid.grid_host)
    finally:
        _close(window)


def test_a_tech_the_starting_age_has_gets_a_red_cell_and_leaves_the_grid(monkeypatch) -> None:
    from descape import civ_availability

    monkeypatch.setattr(civ_availability, "player_start_age", lambda _loaded, player: 5 if player == 1 else None)
    window, _indices = _grid_window(
        _CHANGE_RESEARCH,
        {"technology": _LOOM, "object_list_unit_id_2": _BARRACKS, "button_location": 8, "source_player": 1},
    )
    try:
        grid = _grid(window)
        cell = _cell(grid, 2, 3)
        assert cell.unavailable and cell.frame == "problem" and cell.ghost == (_LOOM, "tech")
        assert "player 1 starts in the Imperial Age, which already has Loom." in cell.toolTip()
        assert "Enable it" not in cell.toolTip(), "nothing gives a researched tech's button back"
        assert (222, "tech") not in _cell(grid, 2, 1).items, "Man-At-Arms is already researched"
    finally:
        _close(window)
