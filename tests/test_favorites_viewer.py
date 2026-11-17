"""GH #195's Favorite Objects window wired into a real offscreen ViewerWindow:
Edit > Favorite Objects…, the unit_favorites keybind, the pick that arms a
placing tool, the gate and hints, and the refreshes. The window on its own is
tests/test_favorites_dialog.py. Every window here closes through
conftest.close_window() (mark_saved() first)."""

from __future__ import annotations

import pytest
from PyQt5.QtCore import Qt

from descape import game_dataset, games, settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_ARCHER = 4
_CASTLE = 82
_VILLAGER = 83
_GATE = 64  # a stone gate orientation: a turn picks a sibling const
_UNKNOWN = 32000


def _units_window(load: bool = True):
    window = conftest.blank_window(load=load)
    window.mode_combo.setCurrentText("Units")
    return window


def _favorites(*ids) -> None:
    settings.set_unit_favorites(game_dataset.active(), list(ids))


def _open(window):
    window.favorites_action.trigger()
    dialog = window._favorites_dialog
    assert dialog is not None and dialog.isVisible()
    return dialog


def _click(dialog, value: int) -> None:
    item = next(dialog.list.item(i) for i in range(dialog.list.count()) if dialog.list.item(i).data(Qt.UserRole) == value)
    dialog.list.setCurrentItem(item)
    dialog.list.itemClicked.emit(item)


def test_the_edit_item_is_live_in_units_mode_only() -> None:
    window = conftest.blank_window()
    try:
        assert window.favorites_action.text() == "Favor&ite Objects…"
        assert window.favorites_action.toolTip()
        edit = next(a.menu() for a in window.menuBar().actions() if a.text().replace("&", "") == "Edit")
        assert window.favorites_action in edit.actions()
        for mode, live in (("Units", True), ("Terrain", False), ("Triggers", False), ("Units", True), ("View", False)):
            window.mode_combo.setCurrentText(mode)
            assert window.favorites_action.isEnabled() == live, mode
    finally:
        conftest.close_window(window)


def test_the_keybind_is_mapped_unbound_and_scoped_to_units() -> None:
    window = conftest.blank_window()
    try:
        assert window._keybind_actions["unit_favorites"] is window.favorites_action
        row = next(row for row in settings.REBINDABLE_ACTIONS if row[0] == "unit_favorites")
        assert row == ("unit_favorites", "Open Favorite Objects", "")
        assert settings._SCOPED_KEYBINDS["unit_favorites"] == frozenset({"units"})
    finally:
        conftest.close_window(window)


def test_opening_pushes_the_rows_the_gate_the_colour_and_the_civ(monkeypatch) -> None:
    _favorites(_CASTLE, _ARCHER, _UNKNOWN)
    window = _units_window()
    try:
        civs = {2: 41}
        monkeypatch.setattr(window.units_panel, "_portrait_civ_for", civs.get)
        assert window.units_panel.select_owner(2)
        dialog = _open(window)
        assert dialog.values() == [_CASTLE, _ARCHER, _UNKNOWN]
        assert dialog.list.item(2).text() == f"Unknown object ({_UNKNOWN})"
        assert dialog.list.isEnabled() and dialog.hint.isHidden()
        assert dialog._team_rgb == window.units_panel._team_rgb(2)
        assert dialog._civ == 41
        again = _open(window)
        assert again is dialog, "one window, raised again"
    finally:
        conftest.close_window(window)


def test_a_catalog_change_refreshes_the_open_window() -> None:
    _favorites(_ARCHER)
    window = _units_window()
    try:
        dialog = _open(window)
        window.units_panel.catalog_view._favorite_menu(_CASTLE).actions()[0].trigger()
        assert dialog.values() == [_ARCHER, _CASTLE]
        window.units_panel.catalog_view.set_favorites([_CASTLE, _ARCHER])
        assert dialog.values() == [_CASTLE, _ARCHER]
    finally:
        conftest.close_window(window)


def test_remove_in_the_window_removes_from_storage_and_the_star() -> None:
    from descape.catalog_browser import FAVORITE_ROLE

    _favorites(_ARCHER, _CASTLE)
    window = _units_window()
    try:
        dialog = _open(window)
        dialog.list.setCurrentRow(0)
        dialog.remove_button.click()
        assert settings.get_unit_favorites(game_dataset.active()) == [_CASTLE]
        assert dialog.values() == [_CASTLE]
        catalog = window.units_panel.catalog_view
        assert not catalog._items[_ARCHER].data(FAVORITE_ROLE)
        dialog._remove_menu(_CASTLE).actions()[0].trigger()
        assert settings.get_unit_favorites(game_dataset.active()) == []
        assert dialog.hint.text() == dialog.EMPTY_HINT
    finally:
        conftest.close_window(window)


def test_a_drop_in_the_window_saves_the_order() -> None:
    _favorites(_ARCHER, _CASTLE, _VILLAGER)
    window = _units_window()
    try:
        dialog = _open(window)
        dialog.list.reordered.emit([_VILLAGER, _ARCHER, _CASTLE])
        assert settings.get_unit_favorites(game_dataset.active()) == [_VILLAGER, _ARCHER, _CASTLE]
        assert dialog.values() == [_VILLAGER, _ARCHER, _CASTLE]
    finally:
        conftest.close_window(window)


def test_a_click_picks_the_const_and_arms_place_unit_from_select(monkeypatch) -> None:
    _favorites(_CASTLE)
    window = _units_window()
    try:
        focused = []
        monkeypatch.setattr(window, "_focus_map_view", lambda: focused.append(True))
        window.select_action.setChecked(True)
        assert window._current_tool == "select"
        dialog = _open(window)
        _click(dialog, _CASTLE)
        assert window.units_panel.selected_object_const() == _CASTLE
        assert window._current_tool == "place_unit" and window.place_unit_action.isChecked()
        assert focused == [True], "the keys go back to the map"
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("tool", ["unit_line", "unit_shape", "unit_brush", "place_unit"])
def test_a_click_keeps_a_placing_tool(tool) -> None:
    _favorites(_ARCHER)
    window = _units_window()
    try:
        window.units_panel.pick_object(_VILLAGER)
        getattr(window, f"{tool}_action").setChecked(True)
        assert window._current_tool == tool
        _click(_open(window), _ARCHER)
        assert window._current_tool == tool
        assert window.units_panel.selected_object_const() == _ARCHER
    finally:
        conftest.close_window(window)


def test_an_unknown_row_picks_nothing() -> None:
    _favorites(_UNKNOWN)
    window = _units_window()
    try:
        before = window.units_panel.selected_object_const()
        _click(_open(window), _UNKNOWN)
        assert window.units_panel.selected_object_const() == before
    finally:
        conftest.close_window(window)


def test_a_refused_pick_logs_and_changes_no_tool(monkeypatch) -> None:
    _favorites(_ARCHER)
    window = _units_window()
    try:
        window.select_action.setChecked(True)
        dialog = _open(window)
        monkeypatch.setattr(window.units_panel, "pick_object", lambda _const: False)
        _click(dialog, _ARCHER)
        assert window._current_tool == "select"
        assert "could not pick" in window.status_log.toPlainText()
        monkeypatch.setattr(window.place_unit_action, "isEnabled", lambda: False)
        window._arm_favorite(_ARCHER)
        assert window._current_tool == "select"
        assert "Place Unit is unavailable" in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)


def test_the_window_greys_out_with_a_hint_outside_units_mode_and_without_a_map(monkeypatch) -> None:
    _favorites(_ARCHER)
    window = _units_window()
    try:
        dialog = _open(window)
        window.mode_combo.setCurrentText("Terrain")
        assert dialog.isVisible() and not dialog.list.isEnabled()
        assert dialog.hint.text() == "Switch to Units mode to place a favorite."
        window.mode_combo.setCurrentText("Units")
        assert dialog.list.isEnabled() and dialog.hint.isHidden()
        monkeypatch.setattr(window, "game_data_lock_reason", lambda: "Shown with the other game's data.")
        window._update_tool_enabled()
        assert not dialog.list.isEnabled() and dialog.hint.text() == "Shown with the other game's data."
    finally:
        conftest.close_window(window)
    empty = _units_window(load=False)
    try:
        empty.favorites_action.trigger()
        dialog = empty._favorites_dialog
        assert not dialog.list.isEnabled() and dialog.hint.text() == "Open a scenario to place a favorite."
    finally:
        conftest.close_window(empty)


def test_a_gate_turn_with_the_view_on_stays_in_it() -> None:
    _favorites(_GATE)
    window = _units_window()
    try:
        catalog = window.units_panel.catalog_view
        catalog.set_favorites_view(True)
        assert window.units_panel.pick_object(_GATE)
        window.place_unit_action.setChecked(True)
        window.rotate_place_ghost(steps=1)
        turned = window.units_panel.selected_object_const()
        assert turned != _GATE, "the turn picked a sibling, or this proves nothing"
        assert catalog.in_favorites_view() and catalog.list.currentItem() is None
    finally:
        conftest.close_window(window)


def test_a_dataset_switch_a_rename_and_a_thumbnail_reset_refresh_it() -> None:
    _favorites(_ARCHER)
    window = _units_window()
    try:
        dialog = _open(window)
        refreshes = (window._refresh_game_lists, window.units_panel.rename_catalog, window.units_panel.refresh_thumbnails)
        for index, refresh in enumerate(refreshes):
            ids = [_ARCHER, _CASTLE][: 1 + index % 2] if index != 2 else [_VILLAGER]
            _favorites(*ids)
            refresh()
            assert dialog.values() == ids, refresh.__name__
    finally:
        conftest.close_window(window)


def test_the_window_goes_with_the_viewer() -> None:
    window = _units_window()
    _open(window)
    conftest.close_window(window)
    assert window._favorites_dialog is None


def test_a_real_game_data_switch_shows_that_games_favorites() -> None:
    here = game_dataset.active()
    other = next(game_id for game_id in games.ids() if game_id != here)
    settings.set_unit_favorites(here, [_ARCHER])
    settings.set_unit_favorites(other, [_VILLAGER, _CASTLE])
    window = _units_window()
    try:
        dialog = _open(window)
        assert dialog.values() == [_ARCHER]
        window.set_game_data(other)
        assert game_dataset.active() == other, "the switch happened, or this proves nothing"
        assert window.units_panel.catalog_view.favorites() == [_VILLAGER, _CASTLE]
        assert dialog.values() == [_VILLAGER, _CASTLE]
        assert dialog.list.item(0).text() == window.units_panel.catalog_view.entry(_VILLAGER).name
    finally:
        window.set_game_data(settings.GAME_DATA_DEFAULT)
        conftest.close_window(window)
    assert game_dataset.active() == here


def test_the_settings_rows_save_and_reach_the_lists(monkeypatch) -> None:
    from PyQt5.QtWidgets import QMenu

    from descape.catalog_browser import FAVORITE_ROLE

    dialog, window = conftest.dialog_and_window()
    try:
        redraws = []
        real = window.refresh_favorite_stars
        monkeypatch.setattr(window, "refresh_favorite_stars", lambda: (redraws.append(True), real()))
        combo = dialog.favorite_star_combo
        assert combo.currentData() == settings.FAVORITE_STAR_STYLE_DEFAULT
        combo.setCurrentIndex(combo.findData("mono"))
        assert settings.get_favorite_star_style() == "mono" and redraws == [True]
        assert "Favorite star: Black and white" in window.status_log.toPlainText()

        assert dialog.right_click_menu_radio.isChecked(), "the default"
        dialog.right_click_toggle_radio.setChecked(True)
        assert settings.get_units_catalog_right_click() == "toggle"
        dialog.right_click_menu_radio.setChecked(True)
        assert settings.get_units_catalog_right_click() == "menu"
        dialog.right_click_toggle_radio.setChecked(True)
        assert settings.get_units_catalog_right_click() == "toggle"
        monkeypatch.setattr(QMenu, "exec_", lambda *_args: (_ for _ in ()).throw(AssertionError("no menu")))
        catalog = window.units_panel.catalog_view
        catalog.resize(380, 400)
        catalog._show_favorite_menu(catalog.list.visualItemRect(catalog.list.item(0)).center())
        first = catalog.list.item(0).data(Qt.UserRole)
        assert catalog.favorites() == [first] and catalog.list.item(0).data(FAVORITE_ROLE)
    finally:
        dialog.close()
        conftest.close_window(window)


def test_the_star_combo_repaints_the_catalog_and_the_open_window(monkeypatch) -> None:
    from descape.settings_dialog import SettingsDialog

    window = _units_window()
    try:
        window.refresh_favorite_stars()  # no window yet: the catalog alone
        dialog_window = _open(window)
        hits = []
        for name, viewport in (("catalog", window.units_panel.catalog_view.list.viewport()), ("window", dialog_window.list.viewport())):
            monkeypatch.setattr(viewport, "update", lambda *_args, n=name: hits.append(n))
        dialog = SettingsDialog(window)
        try:
            dialog.favorite_star_combo.setCurrentIndex(dialog.favorite_star_combo.findData("mono"))
        finally:
            dialog.close()
        assert sorted(hits) == ["catalog", "window"]
    finally:
        conftest.close_window(window)


def test_the_dialog_opens_on_the_stored_star_and_right_click() -> None:
    settings.set_favorite_star_style("mono")
    settings.set_units_catalog_right_click("toggle")
    dialog, window = conftest.dialog_and_window()
    try:
        assert dialog.favorite_star_combo.currentData() == "mono"
        assert dialog.right_click_toggle_radio.isChecked() and not dialog.right_click_menu_radio.isChecked()
        assert settings.get_favorite_star_style() == "mono" and settings.get_units_catalog_right_click() == "toggle"
    finally:
        dialog.close()
        conftest.close_window(window)
