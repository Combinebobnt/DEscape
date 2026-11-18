"""Settings' Apply, OK, Cancel and Reset buttons and the unapplied-changes prompt (GH #191).

Every control still writes config.yaml and shows its change at once; Cancel
puts back what config.yaml and the toolbar's Tool Options row held at open (or
at the last Apply), and Reset puts the current tab on its defaults. Built on
conftest.dialog_and_window(), so never through exec_(); the prompt is patched
where a test reaches it (conftest makes an unpatched one raise).
"""

from __future__ import annotations

import pytest

from descape import game_dataset, grid_overlay, settings, settings_snapshot, themes, tool_options, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture
def dialog_window():
    """A dialog and its window, with the app's theme and font put back afterwards."""
    from descape import gc_hold, viewer_dialogs

    with conftest.restoring_app_theme() as app:
        saved_font, saved_default_font = app.font(), viewer_dialogs._DEFAULT_FONT
        viewer_dialogs._DEFAULT_FONT = None
        dialog, window = conftest.dialog_and_window()
        try:
            yield dialog, window
        finally:
            dialog.close()
            conftest.close_window(window)
            viewer_dialogs._DEFAULT_FONT = saved_default_font
            with gc_hold.paused():
                app.setFont(saved_font)


def _button(dialog, name: str):
    from PyQt5.QtWidgets import QDialogButtonBox

    return dialog._buttons.button(getattr(QDialogButtonBox, name))


def _shown(dialog):
    from PyQt5.QtWidgets import QApplication

    dialog.show()
    QApplication.processEvents()
    return dialog


def _answer(monkeypatch, answer: str) -> list[int]:
    from descape.settings_dialog import SettingsDialog

    asked: list[int] = []
    monkeypatch.setattr(SettingsDialog, "_ask_unapplied", lambda _self: asked.append(1) or answer)
    return asked


def _x(widget, ancestor) -> int:
    from PyQt5.QtCore import QPoint

    return widget.mapTo(ancestor, QPoint(0, 0)).x()


def _tab(dialog, title: str) -> None:
    tabs = dialog._tabs
    tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i) == title))


# -- buttons and Apply's state ------------------------------------------------


def test_buttons_sit_in_kde_order_even_after_a_theme_change(dialog_window) -> None:
    from PyQt5.QtWidgets import QApplication, QDialogButtonBox, QStyle

    dialog, _window = dialog_window
    # The proxy does the work: the app style's own order here is not KDE's.
    assert QApplication.style().styleHint(QStyle.SH_DialogButtonLayout) != QDialogButtonBox.KdeLayout
    _shown(dialog)
    names = ("Reset", "Ok", "Apply", "Cancel")
    assert sorted(names, key=lambda name: _x(_button(dialog, name), dialog)) == list(names)

    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("dark"))
    QApplication.processEvents()
    assert sorted(names, key=lambda name: _x(_button(dialog, name), dialog)) == list(names)
    dialog._cancel()


def test_the_prompt_reads_apply_discard_back_with_back_the_default(dialog_window) -> None:
    from PyQt5.QtWidgets import QDialogButtonBox

    dialog, _window = dialog_window
    box, apply_button, discard_button = dialog._unapplied_box()
    back_button = box.defaultButton()
    assert back_button.text() == "Back" and box.escapeButton() is back_button
    # The button row's layout order, never shown: show() on this QMessageBox hung offscreen on Windows CI.
    row = box.findChild(QDialogButtonBox).layout()
    assert row.indexOf(apply_button) < row.indexOf(discard_button) < row.indexOf(back_button)
    box.done(0)


def test_apply_is_enabled_only_while_something_is_unapplied(dialog_window) -> None:
    dialog, _window = dialog_window
    apply_button = _button(dialog, "Apply")
    assert not apply_button.isEnabled()

    dialog.zoom_view_center_radio.setChecked(True)
    assert apply_button.isEnabled()
    apply_button.click()
    assert not apply_button.isEnabled()
    assert settings.get_zoom_centered_on_cursor() is False

    dialog.zoom_cursor_radio.setChecked(True)
    assert apply_button.isEnabled()
    dialog.zoom_view_center_radio.setChecked(True)
    assert not apply_button.isEnabled(), "back to what was applied"

    # A keyboard slider change waits on its debounce, and counts as unapplied meanwhile.
    dialog.pan_speed_slider.setValue(dialog.pan_speed_slider.value() + 100)
    assert dialog._pan_speed_apply_timer.isActive() and apply_button.isEnabled()
    dialog._pan_speed_apply_timer.stop()
    dialog._persist_pan_speed()
    dialog._refresh_apply()
    assert apply_button.isEnabled(), "now saved, and still unapplied"
    dialog._apply()
    assert not apply_button.isEnabled()


# -- Cancel -------------------------------------------------------------------


def test_cancel_restores_theme_font_and_keybind(dialog_window) -> None:
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication, QDialog

    from descape.viewer_dialogs import apply_theme

    dialog, window = dialog_window
    app = QApplication.instance()
    apply_theme(app, settings.get_theme(), settings.get_theme_colors())  # as main() does at startup
    window_colour = app.palette().color(QPalette.Window).name()
    point_size = app.font().pointSize()
    action = window._keybind_actions["file_open"]
    shortcut = action.shortcut().toString()

    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("dark"))
    assert app.palette().color(QPalette.Window).name() != window_colour
    new_size = settings.UI_FONT_SIZE_MAX if point_size != settings.UI_FONT_SIZE_MAX else settings.UI_FONT_SIZE_MIN
    dialog.ui_font_size_spin.setValue(new_size)
    dialog._flush_pending()
    assert app.font().pointSize() == new_size
    from PyQt5.QtGui import QKeySequence

    dialog._keybind_edits["file_open"].setKeySequence(QKeySequence("Ctrl+F12"))
    assert action.shortcut().toString() == "Ctrl+F12"

    _button(dialog, "Cancel").click()

    assert dialog.result() == QDialog.Rejected
    assert settings.get_theme() == themes.THEME_DEFAULT
    assert settings.get_ui_font_size() is None
    assert settings.get_keybind("file_open") == "Ctrl+O"
    assert app.palette().color(QPalette.Window).name() == window_colour
    assert app.font().pointSize() == point_size
    assert action.shortcut().toString() == shortcut
    assert settings._SAVE_LISTENERS == []


def test_a_font_changed_and_back_inside_its_debounce_is_shown_right_after_cancel(dialog_window) -> None:
    from PyQt5.QtWidgets import QApplication

    dialog, _window = dialog_window
    app = QApplication.instance()
    start = dialog.ui_font_size_spin.value()
    other = start + 1 if start < settings.UI_FONT_SIZE_MAX else start - 1
    dialog.ui_font_size_spin.setValue(other)
    dialog._flush_pending()
    assert app.font().pointSize() == other
    dialog.ui_font_size_spin.setValue(start)
    assert dialog._ui_font_apply_timer.isActive()
    dialog._cancel()
    assert app.font().pointSize() == start


def test_a_failing_applier_still_runs_the_rest_and_closes(dialog_window, monkeypatch) -> None:
    from descape.settings_dialog import SettingsDialog

    dialog, window = dialog_window
    _shown(dialog)
    row_before = window.tool_option_values()
    dialog.tool_defaults_page.controls["brush_size"].setValue(row_before["brush_size"] + 2)
    dialog.zoom_view_center_radio.setChecked(True)
    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("dark"))

    def broken(_dialog):
        raise RuntimeError("redraw failed")

    monkeypatch.setattr(SettingsDialog, "_apply_zoom_anchor", broken)
    with pytest.raises(RuntimeError, match="redraw failed"):
        dialog._cancel()
    assert settings.get_theme() == themes.THEME_DEFAULT
    assert window.tool_option_values() == row_before, "the toolbar row is put back too"
    assert not dialog.isVisible()
    assert settings._SAVE_LISTENERS == []


def test_cancel_undoes_a_theme_import(dialog_window) -> None:
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_dialogs import apply_theme

    dialog, _window = dialog_window
    app = QApplication.instance()
    apply_theme(app, settings.get_theme(), settings.get_theme_colors())
    window_colour = app.palette().color(QPalette.Window).name()
    opened = settings_snapshot.snapshot_settings(settings_snapshot.ALL_DIALOG_KEYS)
    dialog._apply_imported_appearance(
        {
            "theme": {"preset": "dark", "colors": {"highlight": "#123456"}},
            "overlay_colors": {"ruler_line": "#00ff00"},
            "grid_thickness": 4,
            "trigger_status_marker": "icon",
        }
    )
    assert app.palette().color(QPalette.Window).name() != window_colour
    dialog._cancel()
    assert settings_snapshot.snapshot_settings(settings_snapshot.ALL_DIALOG_KEYS) == opened
    assert app.palette().color(QPalette.Window).name() == window_colour
    assert settings.get_grid_thickness() == grid_overlay.THICKNESS_DEFAULT


def test_cancel_after_apply_undoes_only_the_later_change(dialog_window) -> None:
    dialog, _window = dialog_window
    dialog.zoom_view_center_radio.setChecked(True)
    dialog._apply()
    dialog.backups_enabled_check.setChecked(not dialog.backups_enabled_check.isChecked())
    backups = settings.get_backups_enabled()
    dialog._cancel()
    assert settings.get_zoom_centered_on_cursor() is False
    assert settings.get_backups_enabled() is not backups


def test_cancel_redraws_only_for_what_changed(dialog_window, monkeypatch) -> None:
    dialog, window = dialog_window
    calls: list[str] = []
    monkeypatch.setattr(window, "refresh_map", lambda: calls.append("refresh_map"))
    monkeypatch.setattr(window, "apply_game_data", lambda: calls.append("apply_game_data"))
    dialog.zoom_view_center_radio.setChecked(True)
    dialog._cancel()
    assert calls == [], "zoom alone redraws nothing"
    assert settings.get_zoom_centered_on_cursor() is True


def test_a_cancelled_graphics_quality_redraws_and_restores_the_hi_res_flags(dialog_window, monkeypatch) -> None:
    dialog, window = dialog_window
    calls: list[str] = []
    monkeypatch.setattr(window, "refresh_map", lambda: calls.append("refresh_map"))
    assert not unit_sprites.hi_res()
    dialog.graphics_quality_slider.setValue(settings.GRAPHICS_QUALITY_MAX)
    dialog._flush_pending()
    assert unit_sprites.hi_res() and calls == ["refresh_map"]
    dialog._cancel()
    assert settings.get_graphics_quality() == settings.GRAPHICS_QUALITY_DEFAULT
    assert not unit_sprites.hi_res()
    assert calls == ["refresh_map", "refresh_map"]


def test_an_opacity_drag_cancelled_inside_its_debounce_leaves_the_map(dialog_window, monkeypatch) -> None:
    dialog, window = dialog_window
    applied: list[dict | None] = []
    real = window.map_view.apply_overlay_opacity
    monkeypatch.setattr(
        window.map_view, "apply_overlay_opacity", lambda overrides=None: applied.append(overrides) or real(overrides)
    )
    slider = dialog._overlay_opacity_sliders["region"]
    slider.setValue(slider.value() - 30)
    assert dialog._overlay_opacity_timer.isActive()
    assert applied[-1] == {"region": slider.value()}
    dialog._cancel()
    assert not dialog._overlay_opacity_timer.isActive()
    assert applied[-1] is None, "the map is back on the saved opacity"
    assert settings.get_overlay_opacity("region") == settings.get_default_overlay_opacity("region")


def test_cancel_restores_the_mods_tab(dialog_window) -> None:
    from descape import mod_graphics

    dialog, _window = dialog_window
    settings.set_mod_art_excluded({"/mods/a/status"})  # what unticking a listed mod writes
    dialog._apply_mod_art()
    assert mod_graphics._excluded == frozenset({"/mods/a/status"})
    dialog.data_mod_label_check.setChecked(not settings.get_data_mod_label_shown())
    dialog._cancel()
    assert settings.get_mod_art_excluded() == frozenset()
    assert mod_graphics._excluded == frozenset()
    assert settings.get_data_mod_label_shown() is True


def test_rescan_mods_is_not_undone_by_cancel(dialog_window, monkeypatch) -> None:
    dialog, _window = dialog_window
    restored: list[int] = []
    monkeypatch.setattr(settings_snapshot, "restore_settings", lambda _snapshot: restored.append(1))
    dialog.rescan_mods()
    dialog._cancel()
    assert restored == []


def test_cancel_restores_a_tools_default_and_the_toolbar_row(dialog_window) -> None:
    dialog, window = dialog_window
    page = dialog.tool_defaults_page
    row_before = window.tool_option_values()
    page.controls["brush_size"].setValue(row_before["brush_size"] + 2)
    assert settings.get_tool_option_default("brush_size") == row_before["brush_size"] + 2
    assert window.tool_option_values()["brush_size"] == row_before["brush_size"] + 2, "the row follows at once"
    dialog._cancel()
    assert settings.get_tool_option_default("brush_size") == tool_options.default("brush_size")
    assert window.tool_option_values() == row_before


def test_cancel_puts_back_a_legacy_tools_key(tmp_path) -> None:
    import yaml

    settings.CONFIG_PATH.write_text(yaml.safe_dump({"paint_trees": False}))
    dialog, window = conftest.dialog_and_window()
    try:
        assert settings.get_tool_option_default("paint_trees") is False
        dialog.tool_defaults_page.controls["brush_size"].setValue(3)
        assert "paint_trees" not in yaml.safe_load(settings.CONFIG_PATH.read_text())
        dialog._cancel()
        assert yaml.safe_load(settings.CONFIG_PATH.read_text()) == {"paint_trees": False}
        assert settings.get_tool_option_default("paint_trees") is False
    finally:
        dialog.close()
        conftest.close_window(window)


def test_cancel_restores_game_data_and_the_toolbar_wall_pick(dialog_window, monkeypatch) -> None:
    dialog, window = dialog_window
    window.load_scenario(BLANK_TEMPLATE_PATH)
    row_before = window.tool_option_values()
    dialog.game_data_combo.setCurrentIndex(dialog.game_data_combo.findData(game_dataset.ROR))
    assert game_dataset.active() == game_dataset.ROR
    # What a switch can drop (a wall or cliff pick the other game lacks), here made by hand.
    window.tool_option_bindings["brush_size"].apply(row_before["brush_size"] + 1)
    assert window.tool_option_values() != row_before
    calls: list[str] = []
    real = window.apply_game_data
    monkeypatch.setattr(window, "apply_game_data", lambda: calls.append("apply_game_data") or real())
    dialog._cancel()
    assert calls == ["apply_game_data"]
    assert settings.get_game_data() == settings.GAME_DATA_DEFAULT
    assert game_dataset.active() == game_dataset.AOE2
    assert window.tool_option_values() == row_before


def test_cancel_with_nothing_changed_writes_nothing(dialog_window, monkeypatch) -> None:
    dialog, _window = dialog_window
    saved: list[int] = []
    monkeypatch.setattr(settings, "_save_config", lambda _config: saved.append(1))
    dialog._cancel()
    assert saved == []


# -- Esc, the title-bar X, OK -------------------------------------------------


def test_escape_with_changes_asks_once_and_back_keeps_them_live(dialog_window, monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    dialog, _window = dialog_window
    _shown(dialog)
    dialog.zoom_view_center_radio.setChecked(True)
    asked = _answer(monkeypatch, "back")
    QTest.keyClick(dialog, Qt.Key_Escape)
    assert asked == [1]
    assert dialog.isVisible()
    assert settings.get_zoom_centered_on_cursor() is False

    _answer(monkeypatch, "discard")
    dialog.close()  # the title-bar X
    assert not dialog.isVisible()
    assert settings.get_zoom_centered_on_cursor() is True
    assert settings._SAVE_LISTENERS == []


def test_x_with_changes_then_apply_keeps_them(dialog_window, monkeypatch) -> None:
    from PyQt5.QtWidgets import QDialog

    dialog, _window = dialog_window
    _shown(dialog)
    dialog.zoom_view_center_radio.setChecked(True)
    asked = _answer(monkeypatch, "apply")
    dialog.close()
    assert asked == [1]
    assert not dialog.isVisible() and dialog.result() == QDialog.Accepted
    assert settings.get_zoom_centered_on_cursor() is False


def test_no_prompt_without_changes_and_never_from_cancel(dialog_window, monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    dialog, _window = dialog_window
    asked = _answer(monkeypatch, "back")
    _shown(dialog)
    QTest.keyClick(dialog, Qt.Key_Escape)
    assert asked == [] and not dialog.isVisible()

    _shown(dialog)
    dialog.zoom_view_center_radio.setChecked(True)
    _button(dialog, "Cancel").click()
    assert asked == [] and not dialog.isVisible()
    assert settings.get_zoom_centered_on_cursor() is True


def test_ok_keeps_the_changes(dialog_window) -> None:
    from PyQt5.QtWidgets import QDialog

    dialog, _window = dialog_window
    _shown(dialog)
    dialog.zoom_view_center_radio.setChecked(True)
    dialog.pan_speed_slider.setValue(dialog.pan_speed_slider.value() + 100)
    _button(dialog, "Ok").click()
    assert dialog.result() == QDialog.Accepted
    assert settings.get_zoom_centered_on_cursor() is False
    assert settings.get_pan_speed() == dialog.pan_speed_slider.value(), "a pending debounce is applied too"
    assert settings._SAVE_LISTENERS == []


def test_a_never_shown_close_keeps_changes_and_stops_listening() -> None:
    dialog, window = conftest.dialog_and_window()
    try:
        dialog.zoom_view_center_radio.setChecked(True)
        dialog.close()
        assert settings.get_zoom_centered_on_cursor() is False
        assert settings._SAVE_LISTENERS == []
    finally:
        conftest.close_window(window)


def test_return_in_a_field_does_not_close_the_dialog(dialog_window) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    dialog, _window = dialog_window
    _shown(dialog)
    for edit in (dialog.install_path_edit, dialog._keybind_filter_edit):
        edit.setFocus()
        QTest.keyClick(edit, Qt.Key_Return)
        assert dialog.isVisible(), edit
    _button(dialog, "Ok").setFocus()
    QTest.keyClick(_button(dialog, "Ok"), Qt.Key_Return)
    assert not dialog.isVisible(), "a focused button still takes Return"


def test_quitting_for_an_import_closes_without_asking(dialog_window, monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication, QDialog

    dialog, window = dialog_window
    closed: list[int] = []
    monkeypatch.setattr(window, "close", lambda: closed.append(1))
    _shown(dialog)
    dialog.zoom_view_center_radio.setChecked(True)
    dialog._quit_for_import()
    QApplication.processEvents()
    assert closed == [1]
    assert not dialog.isVisible() and dialog.result() == QDialog.Accepted
    assert settings.get_zoom_centered_on_cursor() is False


# -- Reset --------------------------------------------------------------------


def test_reset_general_keeps_the_install_path(dialog_window, tmp_path) -> None:
    import yaml

    from descape import asset_source

    dialog, _window = dialog_window
    asset_source.save_install_path_to_config(tmp_path)
    dialog.zoom_view_center_radio.setChecked(True)
    _tab(dialog, "General")
    _button(dialog, "Reset").click()
    assert settings.get_zoom_centered_on_cursor() is True
    assert dialog.zoom_cursor_radio.isChecked()
    assert yaml.safe_load(settings.CONFIG_PATH.read_text())["aoe2de_install"] == str(tmp_path)
    assert "install path is not reset" in _button(dialog, "Reset").toolTip()


def test_reset_mods_keeps_the_profile_folder(dialog_window, tmp_path) -> None:
    import yaml

    from descape import asset_source

    dialog, _window = dialog_window
    asset_source.save_profile_path_to_config(tmp_path)
    dialog.data_mod_hide_unchanged_check.setChecked(True)
    _tab(dialog, "Mods")
    _button(dialog, "Reset").click()
    assert settings.get_data_mod_hide_unchanged() is False
    assert not dialog.data_mod_hide_unchanged_check.isChecked()
    assert yaml.safe_load(settings.CONFIG_PATH.read_text())["aoe2de_profile"] == str(tmp_path)
    assert "profile folder is not reset" in _button(dialog, "Reset").toolTip()


def test_every_key_reset_keeps_has_its_tabs_tooltip_note() -> None:
    from descape import settings_snapshot
    from descape.settings_dialog_apply import _SettingsDialogApplyMixin

    for key in settings_snapshot.RESET_KEEPS:
        tab = next(tab for tab, keys in settings_snapshot.DIALOG_TAB_KEYS.items() if key in keys)
        assert tab in _SettingsDialogApplyMixin._RESET_KEEPS, (key, tab)


def test_reset_saving_keeps_the_autosave_folder_and_cancel_still_restores_it(dialog_window, tmp_path) -> None:
    dialog, _window = dialog_window
    folder = tmp_path / "autosaves"
    folder.mkdir()
    settings.set_autosave_dir(str(folder))
    backups = settings.get_backups_enabled()
    dialog._apply()
    dialog.backups_enabled_check.setChecked(not backups)
    _tab(dialog, "Saving")
    _button(dialog, "Reset").click()
    assert settings.get_backups_enabled() == backups
    assert settings.get_autosave_dir() == str(folder)
    assert "autosave folder is not reset" in _button(dialog, "Reset").toolTip()
    settings.set_autosave_dir("")
    dialog._cancel()
    assert settings.get_autosave_dir() == str(folder)


def test_reset_appearance_leaves_other_tabs_and_cancel_undoes_it(dialog_window) -> None:
    dialog, _window = dialog_window
    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("dark"))
    dialog.zoom_view_center_radio.setChecked(True)
    dialog._apply()
    dialog.backups_enabled_check.setChecked(not settings.get_backups_enabled())
    backups = settings.get_backups_enabled()
    old_combo = dialog.theme_combo
    _tab(dialog, "Appearance")
    _button(dialog, "Reset").click()
    assert settings.get_theme() == themes.THEME_DEFAULT
    assert dialog.theme_combo is not old_combo, "the page was rebuilt"
    assert dialog.theme_combo.currentData() == settings.get_theme()
    assert dialog._tabs.tabText(dialog._tabs.currentIndex()) == "Appearance"
    assert settings.get_zoom_centered_on_cursor() is False and settings.get_backups_enabled() is backups
    assert _button(dialog, "Apply").isEnabled()

    # The rebuilt page still edits.
    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("light"))
    assert settings.get_theme() == "light"
    dialog.ruler_label_font_spin.setValue(dialog.ruler_label_font_spin.value() + 1)
    assert settings.get_ruler_label_font_px() == dialog.ruler_label_font_spin.value()

    dialog._cancel()
    assert settings.get_theme() == "dark"
    assert settings.get_backups_enabled() is not backups


def test_reset_tools_matches_reset_all_to_defaults(dialog_window) -> None:
    dialog, window = dialog_window
    page = dialog.tool_defaults_page
    page.controls["brush_size"].setValue(4)
    window.tool_option_bindings["auto_beach"].apply(not tool_options.default("auto_beach"))
    _tab(dialog, "Tools")
    _button(dialog, "Reset").click()
    assert settings.get_tool_option_defaults() == tool_options.defaults()
    assert page.controls["brush_size"].value() == tool_options.default("brush_size")
    reset_row = window.tool_option_values()

    page.controls["brush_size"].setValue(4)
    page.reset_all_button.click()
    assert window.tool_option_values() == reset_row


def test_reset_keybinds_restores_the_default_shortcuts(dialog_window) -> None:
    from PyQt5.QtGui import QKeySequence

    dialog, window = dialog_window
    dialog._keybind_edits["file_open"].setKeySequence(QKeySequence("Ctrl+F12"))
    dialog._wheel_combos["plain"].setCurrentIndex(settings.WHEEL_ACTIONS.index("none"))
    _tab(dialog, "Keybinds")
    _button(dialog, "Reset").click()
    assert window._keybind_actions["file_open"].shortcut().toString() == "Ctrl+O"
    assert dialog._keybind_edits["file_open"].keySequence().toString() == "Ctrl+O"
    assert settings.get_wheel_bindings() == settings.DEFAULT_WHEEL_BINDINGS
    _button(dialog, "Ok").click()
    assert settings.get_keybind("file_open") == "Ctrl+O"


def test_reset_advanced_gives_auto_and_hide(dialog_window) -> None:
    dialog, _window = dialog_window
    dialog.game_data_combo.setCurrentIndex(dialog.game_data_combo.findData(game_dataset.ROR))
    dialog.gaia_only_keep_radio.setChecked(True)
    _tab(dialog, "Advanced")
    _button(dialog, "Reset").click()
    assert settings.get_game_data() == settings.GAME_DATA_DEFAULT
    assert settings.get_gaia_only_catalog_mode() == "hide"
    assert game_dataset.active() == game_dataset.AOE2
    assert dialog.game_data_combo.currentData() == settings.GAME_DATA_DEFAULT
    assert dialog.gaia_only_hide_radio.isChecked()


def test_reset_saving_rebuilds_and_cancel_brings_it_back(dialog_window) -> None:
    dialog, _window = dialog_window
    dialog.autosave_interval_combo.setCurrentIndex(0)
    minutes = settings.get_autosave_interval_min()
    dialog._apply()
    _tab(dialog, "Saving")
    _button(dialog, "Reset").click()
    assert settings.get_autosave_interval_min() == settings.AUTOSAVE_INTERVAL_DEFAULT
    assert dialog.autosave_interval_combo.currentData() == settings.AUTOSAVE_INTERVAL_DEFAULT
    dialog._cancel()
    assert settings.get_autosave_interval_min() == minutes


# -- structure ----------------------------------------------------------------


def test_every_dialog_key_is_applied_or_read_where_used() -> None:
    from descape.settings_dialog_apply import _SettingsDialogApplyMixin as Mixin

    applied = [key for _name, keys in Mixin._APPLIERS for key in keys]
    covered = applied + list(Mixin._READ_WHEN_USED)
    assert sorted(covered) == sorted(settings_snapshot.ALL_DIALOG_KEYS)
    from descape.settings_dialog import SettingsDialog

    for name, _keys in Mixin._APPLIERS:
        assert callable(getattr(SettingsDialog, name, None)), name


def test_the_mixin_follows_the_split_rules() -> None:
    from PyQt5.QtWidgets import QDialog

    from descape.settings_dialog import SettingsDialog
    from descape.settings_dialog_apply import _SettingsDialogApplyMixin as Mixin

    bases = SettingsDialog.__bases__
    assert bases.index(Mixin) < bases.index(QDialog)
    own = {name for name in vars(Mixin) if not name.startswith("__")}
    assert not own & set(vars(SettingsDialog)), "a name defined in both"
    assert "__init__" not in vars(Mixin)
    assert not [name for name in own if hasattr(QDialog, name)], "a Qt override or signal in the mixin"
