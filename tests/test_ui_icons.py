"""TASK-271: every tool and menu command wears an icon, checkable toggles keep their
checkmark instead, a theme switch re-inks them, and Settings > Appearance's Toolbar
buttons picks the tool button style without losing a tool to the overflow.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
Pixel reads are PyQt 5.15, Fusion, offscreen QPA.
"""

from __future__ import annotations

import pytest

from descape import settings, viewer_common

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _icon_problems(window) -> list[str]:
    """Every menu bar menu plus the Filters and More Tools popups."""
    menus = [(top.text(), top.menu()) for top in window.menuBar().actions() if top.menu() is not None]
    menus += [("Filters", window.filters_button.menu()), ("More Tools", window.more_tools_menu)]
    problems = []
    for title, menu in menus:
        for where, action in conftest.menu_items(menu, title):
            has_icon = not action.icon().isNull() and action.isIconVisibleInMenu()
            # Open Recent's file entries carry their path as data; the placeholder is disabled.
            recent_entry = action in window.recent_menu.actions() and action is not window.clear_recent_action
            bare = action.menu() is not None or action.isCheckable() or recent_entry
            if bare and has_icon:
                problems.append(f"{where}: has an icon")
            elif not bare and not has_icon:
                problems.append(f"{where}: no icon")
    return problems


def test_every_menu_command_has_an_icon_and_no_toggle_does():
    window = conftest.blank_window(load=False)
    try:
        assert _icon_problems(window) == []
    finally:
        conftest.close_window(window)


def test_clear_recent_files_gets_its_icon_on_every_rebuild(tmp_path):
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.blank_window(load=False)
    try:
        assert window.clear_recent_action is None
        settings.add_recent_file(BLANK_TEMPLATE_PATH)
        window._rebuild_recent_files_menu()
        assert window.clear_recent_action is not None
        assert not window.clear_recent_action.icon().isNull()
        assert _icon_problems(window) == []
        window._clear_recent_files()
        assert window.clear_recent_action is None
    finally:
        conftest.close_window(window)


def test_every_tool_action_has_an_icon_and_a_named_tooltip():
    window = conftest.blank_window(load=False)
    try:
        for tool in settings.TOOLS:
            action = getattr(window, f"{tool.tool_id}_action")
            assert not action.icon().isNull(), tool.tool_id
            assert action.toolTip().startswith(f"{tool.label}"), (tool.tool_id, action.toolTip())
        key = settings.get_keybind("tool_pan")
        assert key, "Pan has a default key"
        assert window.pan_action.toolTip().startswith(f"Pan ({key}): ")
    finally:
        conftest.close_window(window)


def test_a_rebound_tool_key_shows_in_its_tooltip_once():
    window = conftest.blank_window(load=False)
    try:
        before = window.draw_action.toolTip()
        settings.set_keybind("tool_draw", "Ctrl+J")
        window.apply_keybind("tool_draw")
        after = window.draw_action.toolTip()
        assert after.startswith("Draw (Ctrl+J): ")
        assert after.split(": ", 1)[1] == before.split(": ", 1)[1]
    finally:
        conftest.close_window(window)


def _icon_bytes(action) -> bytes:
    image = action.icon().pixmap(24, 24).toImage()
    return bytes(image.constBits().asarray(image.sizeInBytes()))


def test_a_theme_switch_re_inks_the_icons():
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_dialogs import apply_theme

    with conftest.restoring_app_theme() as app:
        window = conftest.blank_window(load=False)
        try:
            undo = window._keybind_actions["edit_undo"]
            apply_theme(app, "light", {})
            QApplication.processEvents()
            light = _icon_bytes(window.pan_action), _icon_bytes(undo)
            apply_theme(app, "dark", {})
            QApplication.processEvents()
            dark = _icon_bytes(window.pan_action), _icon_bytes(undo)
            assert light[0] != dark[0]
            assert light[1] != dark[1]
        finally:
            conftest.close_window(window)


_STYLES = {"icons": "ToolButtonIconOnly", "icons_text": "ToolButtonTextBesideIcon", "text": "ToolButtonTextOnly"}


@pytest.mark.parametrize("style_id", list(_STYLES))
def test_the_toolbar_style_setting_sets_every_button(style_id):
    from PyQt5.QtCore import Qt

    settings.set_toolbar_style(style_id)
    window = conftest.shown_terrain_window()
    try:
        expected = getattr(Qt, _STYLES[style_id])
        assert window.main_toolbar.toolButtonStyle() == expected
        assert window.filters_button.toolButtonStyle() == expected
        assert window.more_tools_button.toolButtonStyle() == expected
        assert window.main_toolbar.iconSize().width() == 24
    finally:
        conftest.close_window(window)


def _check_partition(window) -> list[str]:
    """Each applicable tool on the bar xor in the menu, the overflow a suffix of TOOLS."""
    in_menu = window.more_tools_menu.actions()
    applicable = [t.tool_id for t in settings.TOOLS if viewer_common.tool_applicable(t.tool_id, "terrain")]
    on_bar = []
    problems = []
    for tool_id in applicable:
        action = getattr(window, f"{tool_id}_action")
        bar = action in window.main_toolbar.actions() and action.isVisible()
        if bar == (action in in_menu):
            problems.append(f"{tool_id}: on bar {bar}, in menu {action in in_menu}")
        if bar:
            on_bar.append(tool_id)
    if on_bar != applicable[: len(on_bar)]:
        problems.append(f"overflow is not a suffix: on bar {on_bar}")
    return problems


def test_switching_styles_live_re_partitions_the_overflow():
    from PyQt5.QtWidgets import QApplication

    settings.set_toolbar_style("icons")
    window = conftest.shown_terrain_window(width=settings.MIN_WINDOW_WIDTH)
    try:
        icon_overflow = len(window.more_tools_menu.actions())
        assert _check_partition(window) == []

        settings.set_toolbar_style("text")
        window._apply_toolbar_style()
        QApplication.processEvents()
        assert _check_partition(window) == []
        text_overflow = len(window.more_tools_menu.actions())
        assert text_overflow > icon_overflow, "text buttons are wider, so more tools overflow"

        settings.set_toolbar_style("icons")
        window._apply_toolbar_style()
        QApplication.processEvents()
        assert _check_partition(window) == []
        assert len(window.more_tools_menu.actions()) == icon_overflow
    finally:
        conftest.close_window(window)


def test_overflowed_tools_keep_their_radio_mark_in_more_tools():
    settings.set_toolbar_style("text")
    window = conftest.shown_terrain_window(width=settings.MIN_WINDOW_WIDTH)
    try:
        assert window.more_tools_menu.actions()
        assert _icon_problems(window) == []
    finally:
        conftest.close_window(window)


def _terrain_partition(window) -> list[bool]:
    return [getattr(window, f"{t.tool_id}_action") in window.more_tools_menu.actions() for t in settings.TOOLS]


def test_a_style_switch_in_units_mode_measures_like_a_fresh_window():
    """The Rotate pair shows only in Units mode; a re-measure there must not count it as pinned width."""
    from PyQt5.QtWidgets import QApplication

    settings.set_toolbar_style("text")
    fresh = conftest.shown_terrain_window(width=1280)
    try:
        expected = _terrain_partition(fresh)
    finally:
        conftest.close_window(fresh)

    settings.set_toolbar_style("icons")
    window = conftest.shown_terrain_window(width=1280)
    try:
        window.mode_combo.setCurrentText("Units")
        QApplication.processEvents()
        assert all(a.isVisible() for a in window._rotate_toolbar_actions), "test assumes the Rotate pair shows"
        settings.set_toolbar_style("text")
        window._apply_toolbar_style()
        window.mode_combo.setCurrentText("Terrain")
        QApplication.processEvents()
        assert any(expected), "test assumes something overflows at 1280 with text buttons"
        assert _terrain_partition(window) == expected
    finally:
        conftest.close_window(window)


def test_the_more_tools_button_wears_the_overflowed_active_tool():
    settings.set_toolbar_style("text")
    window = conftest.shown_terrain_window(width=settings.MIN_WINDOW_WIDTH)
    try:
        default = window.more_tools_button.icon().cacheKey()
        overflowed = next(
            t for t in settings.TOOLS if getattr(window, f"{t.tool_id}_action") in window.more_tools_menu.actions()
        )
        window._on_tool_selected(overflowed.tool_id)
        assert window.more_tools_button.icon().cacheKey() != default
        assert window.more_tools_button.text() == f"{overflowed.label} ▾"
        window._on_tool_selected("pan")
        assert window.more_tools_button.icon().cacheKey() == default
    finally:
        conftest.close_window(window)


def test_the_appearance_combo_applies_the_style_live():
    from PyQt5.QtCore import Qt

    dialog, window = conftest.dialog_and_window()
    try:
        combo = dialog.toolbar_style_combo
        assert combo.currentData() == "icons_text"
        combo.setCurrentIndex(combo.findData("icons"))
        assert settings.get_toolbar_style() == "icons"
        assert window.main_toolbar.toolButtonStyle() == Qt.ToolButtonIconOnly
    finally:
        dialog.close()
        conftest.close_window(window)
