"""Guards that Settings' Cancel and Reset reach every setting the dialog writes (GH #191).

Cancel restores settings_snapshot.DIALOG_TAB_KEYS, so a control that writes a
key the table misses would survive Cancel. Two guards: one drives every
control on each tab and diffs config.yaml, the other reads the source for
every settings setter the dialog can call and checks each against the table.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest
import yaml

from descape import settings, settings_snapshot
from descape.settings_snapshot import ALL_DIALOG_KEYS, DIALOG_TAB_KEYS

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

ROOT = Path(__file__).resolve().parent.parent


def _config() -> dict:
    path = settings.CONFIG_PATH
    return (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}


def _key_sequences():
    from PyQt5.QtGui import QKeySequence

    modifiers = ("Ctrl+Alt+", "Ctrl+Alt+Shift+", "Meta+Ctrl+", "Meta+Alt+", "Meta+Shift+")
    # Cycles: a repeat only auto-clears an earlier row, which writes the same key.
    for modifier, number in itertools.cycle(itertools.product(modifiers, range(1, 36))):
        yield QKeySequence(f"{modifier}F{number}")


def _drive(page) -> None:
    """Moves every value control on `page` off its value. Pickers that open a
    dialog (colour swatches, Browse, Import) and plain buttons are left alone."""
    from PyQt5.QtWidgets import (
        QAbstractButton,
        QCheckBox,
        QComboBox,
        QKeySequenceEdit,
        QRadioButton,
        QSlider,
        QSpinBox,
        QToolButton,
    )

    sequences = _key_sequences()
    moved = set()
    for widget in page.findChildren(QAbstractButton):
        if isinstance(widget, QCheckBox) and widget.isEnabled():
            widget.setChecked(not widget.isChecked())
        elif isinstance(widget, QRadioButton) and not widget.isChecked():
            # One move per group, so a pair flips each drive instead of always landing on its last radio (Windows has no X11 box).
            group = widget.group() or widget.parentWidget()
            if group not in moved:
                moved.add(group)
                widget.setChecked(True)
        elif isinstance(widget, QToolButton) and widget.menu() is not None and widget.menu().actions():
            action = widget.menu().actions()[0]
            action.setChecked(not action.isChecked())
    for combo in page.findChildren(QComboBox):
        if combo.count() > 1 and combo.isEnabled():
            combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())
    for spin in page.findChildren(QSpinBox):
        spin.setValue(spin.value() + 1 if spin.value() < spin.maximum() else spin.value() - 1)
    for slider in page.findChildren(QSlider):
        slider.setValue(slider.value() + 1 if slider.value() < slider.maximum() else slider.value() - 1)
    for edit in page.findChildren(QKeySequenceEdit):
        edit.setKeySequence(next(sequences))


@pytest.fixture
def dialog_window():
    """A dialog on a window with the blank template open (so Game data redraws a map), with the app's
    theme and font put back afterwards."""
    from descape import gc_hold, viewer_dialogs
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    with conftest.restoring_app_theme() as app:
        saved_font, saved_default_font = app.font(), viewer_dialogs._DEFAULT_FONT
        viewer_dialogs._DEFAULT_FONT = None
        dialog, window = conftest.dialog_and_window()
        window.load_scenario(BLANK_TEMPLATE_PATH)
        try:
            yield dialog, window
        finally:
            dialog.close()
            conftest.close_window(window)
            viewer_dialogs._DEFAULT_FONT = saved_default_font
            with gc_hold.paused():
                app.setFont(saved_font)


def _page(dialog, title: str):
    tabs = dialog._tabs
    index = next(i for i in range(tabs.count()) if tabs.tabText(i) == title)
    tabs.setCurrentIndex(index)
    return tabs.widget(index)


def _driven_keys(dialog, title: str, opened: dict) -> set[str]:
    _drive(_page(dialog, title))
    dialog._flush_pending()
    driven = _config()
    return {key for key in opened.keys() | driven.keys() if opened.get(key) != driven.get(key)}


@pytest.mark.parametrize("title", settings_snapshot.TABS)
def test_every_control_on_a_tab_writes_only_its_keys_and_cancel_restores_them(title, dialog_window) -> None:
    dialog, _window = dialog_window
    opened = _config()
    changed = _driven_keys(dialog, title, opened)
    assert changed, f"driving {title} wrote nothing"
    assert changed <= set(DIALOG_TAB_KEYS[title]), changed - set(DIALOG_TAB_KEYS[title])
    dialog._cancel()
    assert _config() == opened


@pytest.mark.parametrize("title", settings_snapshot.TABS)
def test_a_reset_page_still_edits_and_cancel_undoes_both(title, dialog_window) -> None:
    """Reset rebuilds the page (Tools refreshes in place); the old page's widgets are
    deleted, and the new ones must still write their keys and come back on Cancel."""
    from PyQt5.QtCore import QCoreApplication, QEvent

    dialog, _window = dialog_window
    opened = _config()
    _page(dialog, title)
    dialog._reset_tab()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    changed = _driven_keys(dialog, title, opened)
    assert changed <= set(DIALOG_TAB_KEYS[title]), changed - set(DIALOG_TAB_KEYS[title])
    assert _driven_keys(dialog, title, _config()), f"the rebuilt {title} page wrote nothing"
    dialog._cancel()
    assert _config() == opened


# settings function the dialog calls -> the config keys it writes.
_SETTER_KEYS: dict[str, tuple[str, ...]] = {
    "set_zoom_centered_on_cursor": ("zoom_centered_on_cursor",),
    "set_xwayland_on_wayland": ("xwayland_on_wayland",),
    "set_mod_art_excluded": ("mod_art_excluded",),
    "set_mod_art_included": ("mod_art_included",),
    "set_data_mod_hide_unchanged": ("data_mod_hide_unchanged",),
    "set_data_mod_label_shown": ("data_mod_label_shown",),
    "set_data_mod_terrain_group": ("data_mod_terrain_group",),
    "set_theme": ("theme", "theme_colors", "dark_mode"),
    "set_theme_color": ("theme", "theme_colors", "dark_mode"),
    "clear_theme_color": ("theme", "theme_colors", "dark_mode"),
    "apply_appearance_values": DIALOG_TAB_KEYS["Appearance"],
    "set_ui_font_family": ("ui_font_family",),
    "set_ui_font_size": ("ui_font_size",),
    "set_toolbar_style": ("toolbar_style",),
    "set_preload_zoom_levels": ("preload_zoom_levels",),
    "set_graphics_quality": ("graphics_quality", "potato_mode"),
    "set_elev_step_pct": ("elev_step_pct", "elev_step_divisor"),
    "set_pan_speed": ("pan_speed",),
    "set_wall_opacity_pct": ("wall_opacity_pct",),
    "set_overlay_color": ("overlay_colors",),
    "set_overlay_opacity": ("overlay_opacity",),
    "set_ruler_label_font_px": ("ruler_label_font_px",),
    "set_distance_tick_font_px": ("distance_tick_font_px",),
    "set_stack_badge_position": ("stack_badge_position",),
    "set_favorite_star_style": ("favorite_star_style",),
    "set_units_catalog_right_click": ("units_catalog_right_click",),
    "set_terrain_grid_swatch_px": ("terrain_grid_swatch_px",),
    "set_trigger_details_bottom_layout": ("trigger_details_bottom_layout",),
    "set_grid_blend": ("grid_blend", "grid_lightness"),
    "set_grid_thickness": ("grid_thickness",),
    "set_trigger_status_color": ("trigger_status_colors",),
    "set_trigger_status_marker": ("trigger_status_marker",),
    "set_trigger_status_color_ok_rows": ("trigger_status_color_ok_rows",),
    "set_trigger_status_color_sections": ("trigger_status_color_sections",),
    "set_tool_option_defaults": DIALOG_TAB_KEYS["Tools"],
    "reset_tool_option_defaults": DIALOG_TAB_KEYS["Tools"],
    "set_autosave_enabled": ("autosave_enabled",),
    "set_autosave_interval_min": ("autosave_interval_min",),
    "set_autosave_retention": ("autosave_retention",),
    "set_autosave_location": ("autosave_location",),
    "set_autosave_dir": ("autosave_dir",),
    "set_backups_enabled": ("backups_enabled",),
    "set_keybind": ("keybinds",),
    "set_wheel_binding": ("wheel_bindings",),
    "set_game_data": ("game_data",),
    "set_gaia_only_catalog_mode": ("gaia_only_catalog_mode",),
}
# The dialog's whole modules, and the window methods it calls that write settings.
_MODULES = ("descape/settings_dialog.py", "descape/settings_dialog_apply.py", "descape/tool_defaults_page.py")
_WINDOW_METHODS = {
    "descape/viewer_file.py": ("set_game_data", "apply_game_data"),
    "descape/viewer_view.py": ("set_wall_opacity_pct",),
}
_PREFIXES = ("set_", "reset_", "clear_", "apply_")


def _settings_calls(tree: ast.AST) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "settings"
        and node.func.attr.startswith(_PREFIXES)
    }


def _called_setters() -> set[str]:
    called: set[str] = set()
    for rel in _MODULES:
        called |= _settings_calls(ast.parse((ROOT / rel).read_text(encoding="utf-8")))
    for rel, names in _WINDOW_METHODS.items():
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in found} == set(names), rel
        for node in found:
            called |= _settings_calls(node)
    return called


def test_every_setter_the_dialog_calls_writes_only_dialog_keys() -> None:
    called = _called_setters()
    unmapped = called - set(_SETTER_KEYS)
    assert not unmapped, f"map these in _SETTER_KEYS (and their keys in DIALOG_TAB_KEYS): {sorted(unmapped)}"
    for name in called:
        missing = set(_SETTER_KEYS[name]) - set(ALL_DIALOG_KEYS)
        assert not missing, (name, missing)


def test_the_setter_table_has_no_stale_entries() -> None:
    called = _called_setters()
    assert set(_SETTER_KEYS) <= called, sorted(set(_SETTER_KEYS) - called)
    for name in _SETTER_KEYS:
        assert callable(getattr(settings, name, None)), name
