"""Settings' Apply, Cancel and Reset (GH #191), a mixin of SettingsDialog split out of settings_dialog.py.

Every control still writes config.yaml and shows its change at once. The
dialog keeps a snapshot of its keys (settings_snapshot) and of the toolbar's
Tool Options row from when it opened or was last applied: Apply takes new
ones, Cancel puts them back, and Reset drops the current tab's keys. An
applier here only reads settings and pushes the value live, so restoring
config.yaml and running the appliers whose keys changed undoes a change.

The mixin follows viewer.py's split rules: no __init__, no Qt override, no
signal, and no name SettingsDialog defines too. done(), closeEvent() and
reject() stay in settings_dialog.py and call in here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QProxyStyle,
    QSlider,
    QStyle,
    QWidget,
)

from descape import object_catalog, settings, settings_snapshot, tool_options
from descape.viewer_common import _sync_hi_res


class _KdeButtonOrderStyle(QProxyStyle):
    """KDE's button order on every platform: Reset on the left, then OK, Apply, Cancel,
    and the unapplied-changes prompt's Apply, Discard, Back. All else is the base style's."""

    def styleHint(self, hint, option=None, widget=None, return_data=None) -> int:
        if hint == QStyle.SH_DialogButtonLayout:
            return QDialogButtonBox.KdeLayout
        return super().styleHint(hint, option, widget, return_data)


def kde_button_order(box: QDialogButtonBox) -> None:
    """Gives `box` alone the KDE order: a widget's style does not reach its children,
    and apply_theme() resets only the app style, so nothing else changes."""
    style = _KdeButtonOrderStyle(QApplication.style().objectName())
    style.setParent(box)  # setStyle() does not take ownership
    box.setStyle(style)


class _SettingsDialogApplyMixin:
    # Applier -> the config keys it shows. Each runs once, in this order, when any of its keys
    # changed: game data first, since it rebuilds the lists and redraws the map the others touch.
    _APPLIERS: ClassVar[tuple[tuple[str, tuple[str, ...]], ...]] = (
        ("_apply_game_data", ("game_data",)),
        ("_apply_zoom_anchor", ("zoom_centered_on_cursor",)),
        ("_apply_trigger_section_headers", ("trigger_section_headers",)),
        ("_apply_mod_art", ("mod_art_excluded", "mod_art_included")),
        ("_apply_owner_filter", ("data_mod_hide_unchanged", "gaia_only_catalog_mode")),
        ("_apply_data_mod_label", ("data_mod_label_shown",)),
        ("_apply_terrain_group", ("data_mod_terrain_group",)),
        ("_apply_chrome_theme", ("theme", "theme_colors", "dark_mode")),
        ("_apply_ui_font", ("ui_font_family", "ui_font_size")),
        ("_apply_toolbar_buttons", ("toolbar_style",)),
        ("_apply_preload", ("preload_zoom_levels",)),
        ("_apply_map_render", ("graphics_quality", "potato_mode", "elev_step_pct", "elev_step_divisor", "sloped_elev_step_pct")),
        ("_apply_wall_opacity", ("wall_opacity_pct",)),
        ("_apply_overlay_colors", ("overlay_colors",)),
        ("_apply_overlay_opacity", ("overlay_opacity",)),
        ("_apply_map_label_fonts", ("ruler_label_font_px", "distance_tick_font_px")),
        ("_apply_distance_ticks_behind_terrain", ("distance_ticks_behind_terrain",)),
        ("_apply_stack_badge_position", ("stack_badge_position",)),
        ("_apply_favorite_star", ("favorite_star_style",)),
        ("_apply_terrain_grid_swatch", ("terrain_grid_swatch_px",)),
        ("_apply_trigger_details_layout", ("trigger_details_bottom_layout",)),
        ("_apply_grid_appearance", ("grid_blend", "grid_lightness", "grid_thickness")),
        (
            "_apply_trigger_status",
            (
                "trigger_status_colors",
                "trigger_status_marker",
                "trigger_status_color_ok_rows",
                "trigger_status_color_sections",
            ),
        ),
        ("_apply_trigger_details", ("trigger_details",)),
        ("_apply_tool_defaults", ("tool_options", *(key for key, _name in settings._LEGACY_TOOL_OPTION_KEYS))),
        ("_apply_autosave_timer", ("autosave_enabled", "autosave_interval_min")),
        ("_apply_keybinds", ("keybinds",)),
        ("_apply_wheel_bindings", ("wheel_bindings",)),
    )
    # Read where they are used (or at the next launch), so restoring config.yaml is all they need.
    _READ_WHEN_USED: ClassVar[tuple[str, ...]] = (
        "xwayland_on_wayland",
        "pan_speed",
        "autosave_retention",
        "autosave_location",
        "autosave_dir",
        "backups_enabled",
        "units_catalog_right_click",
        "trigger_rename_double_click",
    )
    # What a tab's Reset leaves alone, for its tooltip.
    _RESET_KEEPS: ClassVar[dict[str, str]] = {
        "General": " The install path is not reset.",
        "Mods": " The profile folder is not reset.",
        "Saving": " The autosave folder is not reset.",
    }
    # True while Cancel puts values back: an applier then logs no line that reads as an apply.
    _quiet_appliers = False

    # -- pages and timers --------------------------------------------------

    def _wrapped_page(self, page: QWidget, wrap: str) -> QWidget:
        """`page` as its tab shows it: "see_through" or "scroll" wraps it in a scroll area, "" leaves it bare."""
        if not wrap:
            return page
        return self._scrolled_tab(page, see_through=wrap == "see_through")

    def _make_apply_timers(self) -> None:
        """The debounce timers, made once in __init__ so Reset can rebuild a page under them."""
        self._ui_font_apply_timer = self._debounce_timer(self._apply_ui_font)
        self._graphics_quality_apply_timer = self._debounce_timer(self._persist_graphics_quality)
        self._elev_step_apply_timer = self._debounce_timer(self._persist_elev_step_pct)
        self._sloped_elev_step_apply_timer = self._debounce_timer(self._persist_sloped_elev_step_pct)
        self._pan_speed_apply_timer = self._debounce_timer(self._persist_pan_speed)
        self._wall_opacity_apply_timer = self._debounce_timer(self._persist_wall_opacity)
        # One shared debounce, for pan_speed_slider's reason: every setter is a full YAML round trip.
        self._overlay_opacity_timer = self._debounce_timer(self._persist_overlay_opacity)
        self._grid_apply_timer = self._debounce_timer(self._end_grid_preview)

    def _debounce_timer(self, slot: Callable[[], None]) -> QTimer:
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(slot)
        # A debounce that ends where it started saves nothing, so Apply is re-checked here too.
        timer.timeout.connect(self._refresh_apply)
        return timer

    def _pending_applies(self) -> tuple[tuple[QTimer, Callable[[], None]], ...]:
        """Each debounce timer whose change is not saved (or not shown) yet, with what saves it."""
        return (
            (self._ui_font_apply_timer, self._apply_ui_font),
            (self._graphics_quality_apply_timer, self._persist_graphics_quality),
            (self._elev_step_apply_timer, self._persist_elev_step_pct),
            (self._sloped_elev_step_apply_timer, self._persist_sloped_elev_step_pct),
            (self._pan_speed_apply_timer, self._persist_pan_speed),
            (self._wall_opacity_apply_timer, self._persist_wall_opacity),
            (self._overlay_opacity_timer, self._persist_overlay_opacity),
        )

    def _flush_pending(self) -> None:
        """Every pending change saved and shown now: Apply, OK, an export or import, a never-shown close."""
        # A keyboard change's timer dies with the dialog, which would leave
        # the grid stuck in its unbaked preview.
        self._end_grid_preview()
        # Applied now, not on a timer that can fire after the window (or a test's settings isolation) is gone.
        for timer, persist in self._pending_applies():
            if timer.isActive():
                timer.stop()
                persist()

    def _stop_pending(self) -> None:
        """Every pending change dropped unsaved: the timers stop, the grid preview ends,
        and an unsaved opacity drag leaves the map."""
        # The UI font is saved at once and only shown on its timer, so it is shown now:
        # Cancel re-applies by saved values, and a font changed and back saves none.
        if self._ui_font_apply_timer.isActive():
            self._ui_font_apply_timer.stop()
            self._apply_ui_font()
        for timer, _persist in self._pending_applies():
            timer.stop()
        self._end_grid_preview()
        self._window.map_view.apply_overlay_opacity()

    # -- buttons -----------------------------------------------------------

    def _connect_buttons(self) -> None:
        """Reset | OK, Apply, Cancel in KDE order, Reset's per-tab tooltip, and Apply's
        enabled state, re-checked on every config.yaml save and slider move."""
        box = self._buttons
        kde_button_order(box)
        box.button(QDialogButtonBox.Ok).clicked.connect(self._ok)
        box.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        box.button(QDialogButtonBox.Cancel).clicked.connect(self._cancel)
        box.button(QDialogButtonBox.Reset).clicked.connect(self._reset_tab)
        self._tabs.currentChanged.connect(self._update_reset_tooltip)
        self._update_reset_tooltip()
        for index in range(self._tabs.count()):
            self._watch_sliders(self._tabs.widget(index))
        listener = self._on_settings_saved
        settings.add_save_listener(listener)
        # A dialog deleted without done() (its window closed under it) must not stay a listener.
        self.destroyed.connect(lambda _obj=None: settings.remove_save_listener(listener))
        self._refresh_apply()

    def _watch_sliders(self, page: QWidget) -> None:
        # A drag or a debounce saves later, so a slider move re-checks Apply itself.
        for slider in page.findChildren(QSlider):
            slider.valueChanged.connect(lambda _value: self._refresh_apply())

    def _update_reset_tooltip(self, _index: int | None = None) -> None:
        title = self._tabs.tabText(self._tabs.currentIndex())
        keeps = self._RESET_KEEPS.get(title, "")
        self._buttons.button(QDialogButtonBox.Reset).setToolTip(f"Reset the {title} tab to its defaults.{keeps}")

    def _on_settings_saved(self, config: dict) -> None:
        self._refresh_apply(config)

    def _refresh_apply(self, config: dict | None = None) -> None:
        self._buttons.button(QDialogButtonBox.Apply).setEnabled(self._is_dirty(config))

    def _is_dirty(self, config: dict | None = None) -> bool:
        """Unapplied changes: a debounce or slider drag still pending, the toolbar row off its
        snapshot, or config.yaml (or `config`, just saved) off its own. A setting that writes a
        whole dict (overlay colours, keybinds) can stay off it after a change and back."""
        if any(timer.isActive() for timer, _persist in self._pending_applies()):
            return True
        if any(slider.isSliderDown() for slider in self.findChildren(QSlider)):
            return True
        if self._window.tool_option_values() != self._toolbar_snapshot:
            return True
        return settings_snapshot.differs(self._snapshot, config)

    def _apply(self) -> None:
        dirty = self._is_dirty()
        self._flush_pending()
        self._snapshot = settings_snapshot.snapshot_settings(settings_snapshot.ALL_DIALOG_KEYS)
        self._toolbar_snapshot = self._window.tool_option_values()
        self._refresh_apply()
        if dirty:
            self._window._log_status("Settings applied")

    def _ok(self) -> None:
        self._apply()
        self.accept()

    def _cancel(self) -> None:
        """The Cancel button, and Discard in the unapplied-changes prompt: never asks."""
        self.done(QDialog.Rejected)

    def _finish(self, result: int) -> None:
        """done()'s part: accepted (OK, a quit for an import) keeps every change, anything else reverts."""
        settings.remove_save_listener(self._on_settings_saved)
        if result == QDialog.Accepted:
            self._flush_pending()
        else:
            self._revert()

    def _close_or_ask(self) -> None:
        """reject()'s part, for Esc and the title-bar X: Cancel, asking first when changes are unapplied."""
        answer = self._ask_unapplied() if self._is_dirty() else "discard"
        if answer == "apply":
            self._ok()
        elif answer == "discard":
            self._cancel()

    def _unapplied_box(self) -> tuple[QMessageBox, QWidget, QWidget]:
        """The unapplied-changes prompt and its Apply and Discard buttons; Back is its default and escape."""
        box = QMessageBox(self)
        box.setWindowTitle("Settings")
        box.setIcon(QMessageBox.Question)
        box.setText("You have settings changes that are not applied.")
        box.setInformativeText("Apply keeps them, Discard reverts them, Back returns to Settings.")
        apply_button = box.addButton("Apply", QMessageBox.ApplyRole)
        discard_button = box.addButton("Discard", QMessageBox.DestructiveRole)
        back_button = box.addButton("Back", QMessageBox.RejectRole)
        box.setDefaultButton(back_button)
        box.setEscapeButton(back_button)
        kde_button_order(box.findChild(QDialogButtonBox))
        return box, apply_button, discard_button

    def _ask_unapplied(self) -> str:
        """The answer: "apply", "discard" or "back". Its own method so offscreen tests can answer it without exec_()."""
        box, apply_button, discard_button = self._unapplied_box()
        box.exec_()
        clicked = box.clickedButton()
        if clicked is apply_button:
            return "apply"
        return "discard" if clicked is discard_button else "back"

    # -- cancel and reset --------------------------------------------------

    def _revert(self) -> None:
        """Cancel: pending changes dropped, then config.yaml and the toolbar row put back as they
        were at open or the last Apply, and every applier whose keys changed run again."""
        self._quiet_appliers, restored = True, False
        try:
            self._stop_pending()
            current = settings_snapshot.snapshot_settings(settings_snapshot.ALL_DIALOG_KEYS)
            restored = current != self._snapshot
            if restored:
                settings_snapshot.restore_settings(self._snapshot)
                self._reapply(current, self._snapshot)
        finally:
            self._quiet_appliers = False
            # A raising applier must not leave the toolbar row on the cancelled values.
            if self._window.tool_option_values() != self._toolbar_snapshot:
                self._restore_tool_row(self._toolbar_snapshot)
                restored = True
            if restored:
                self._window._log_status("Settings changes cancelled")

    def _reset_tab(self) -> None:
        """The current tab's settings back to their defaults, shown at once and pending like
        any change, so Cancel undoes it. Tools also puts the row on DEscape's defaults."""
        index = self._tabs.currentIndex()
        title = self._tabs.tabText(index)
        row = tool_options.defaults() if title == "Tools" else self._window.tool_option_values()
        if title == "General":
            self.trigger_header_settings.detach()  # its page is rebuilt; a focus-loss apply would undo the reset
        self._flush_pending()
        before = settings_snapshot.snapshot_settings(settings_snapshot.ALL_DIALOG_KEYS)
        settings_snapshot.reset_settings(settings_snapshot.reset_keys(title))
        try:
            self._reapply(before, settings_snapshot.snapshot_settings(settings_snapshot.ALL_DIALOG_KEYS))
        finally:
            # A Game data switch drops the toolbar's wall and cliff picks.
            if self._window.tool_option_values() != row:
                self._restore_tool_row(row)
            self._rebuild_tab(index)
            self._refresh_apply()
        self._window._log_status(f"Settings: {title} tab reset to defaults")

    def _rebuild_tab(self, index: int) -> None:
        """A reset tab's page built afresh from settings, then swapped in; Tools refreshes in place."""
        title, build, wrap = self._tab_pages()[index]
        if title == "Tools":
            self.tool_defaults_page._refresh()
            return
        page = self._wrapped_page(build(), wrap)
        self._watch_sliders(page)
        old = self._tabs.widget(index)
        self._tabs.removeTab(index)
        self._tabs.insertTab(index, page, title)
        self._tabs.setCurrentIndex(index)
        old.deleteLater()

    def _reapply(self, before: dict[str, object], after: dict[str, object]) -> None:
        """Runs the appliers whose keys differ between two snapshots, so an untouched setting never re-renders."""
        self._run_appliers({key for key, value in after.items() if before[key] != value})

    def _run_appliers(self, changed: set[str]) -> None:
        """Every applier runs even if one raises, so one failed redraw cannot leave the rest
        showing cancelled values; the first error is raised after the last applier."""
        failed: Exception | None = None
        for name, keys in self._APPLIERS:
            if changed.intersection(keys):
                try:
                    getattr(self, name)()
                except Exception as e:  # noqa: BLE001 -- re-raised below, once the rest have run
                    failed = failed or e
        if failed is not None:
            raise failed

    def _restore_tool_row(self, values: dict[str, object]) -> None:
        self._window.reset_tool_options_to(values)
        self.tool_defaults_page._refresh()

    # -- appliers: read settings, show the value, write nothing ---------------

    def _apply_game_data(self) -> None:
        self._window.apply_game_data()

    def _apply_zoom_anchor(self) -> None:
        self._window.map_view.set_zoom_anchor_mode(settings.get_zoom_centered_on_cursor())

    def _apply_mod_art(self) -> None:
        self._window._apply_mod_art_change("Mod art choices applied", quiet=self._quiet_appliers)

    def _apply_owner_filter(self) -> None:
        self._window.units_panel.refresh_owner_filter()

    def _apply_data_mod_label(self) -> None:
        object_catalog.objects.cache_clear()
        self._window.units_panel.rename_catalog()
        self._window._update_unit_inspector_from_selection()

    def _apply_terrain_group(self) -> None:
        self._window.terrain_panel.reload_catalog()
        if self._window.mode == "terrain":
            self._window._apply_left_pane_floor()

    def _apply_toolbar_buttons(self) -> None:
        self._window._apply_toolbar_style()

    def _apply_preload(self) -> None:
        """Off also cancels every warm in flight and settles a pending placeholder level (TASK-031.25 D7)."""
        if not settings.get_preload_zoom_levels():
            self._window._cancel_warms()
        self._window.sync_canvas_placeholders()

    def _apply_map_render(self) -> None:
        # The hi-res flags follow Graphics quality before the map redraws (GH #241).
        _sync_hi_res()
        self._window.refresh_map()

    def _apply_wall_opacity(self) -> None:
        self._window.set_wall_opacity_pct(settings.get_wall_opacity_pct())

    def _apply_overlay_colors(self) -> None:
        self._window.map_view.apply_overlay_colors()

    def _apply_overlay_opacity(self) -> None:
        self._window.map_view.apply_overlay_opacity()

    def _apply_map_label_fonts(self) -> None:
        self._window.map_view.apply_ruler_label_font()
        self._window.map_view.apply_distance_tick_font()

    def _apply_distance_ticks_behind_terrain(self) -> None:
        self._window.map_view.apply_distance_ticks_behind_terrain()

    def _apply_stack_badge_position(self) -> None:
        self._window.map_view.set_stack_badge_position(settings.get_stack_badge_position())

    def _apply_favorite_star(self) -> None:
        self._window.refresh_favorite_stars()

    def _apply_terrain_grid_swatch(self) -> None:
        self._window.terrain_panel.set_grid_swatch_px(settings.get_terrain_grid_swatch_px())

    def _apply_trigger_details_layout(self) -> None:
        self._window._sync_trigger_detail_layout()

    def _apply_grid_appearance(self) -> None:
        view = self._window.map_view
        blend, thickness = settings.get_grid_blend(), settings.get_grid_thickness()
        self._window._apply_grid_change(lambda: view.set_grid_appearance(blend, thickness))

    def _apply_trigger_section_headers(self) -> None:
        self._window.apply_divider_rule(quiet=self._quiet_appliers)

    def _apply_trigger_status(self) -> None:
        self._window.trigger_panel.apply_status_style()

    def _apply_trigger_details(self) -> None:
        self._window.trigger_panel.refresh_trigger_details()

    def _apply_tool_defaults(self) -> None:
        self.tool_defaults_page._refresh()

    def _apply_autosave_timer(self) -> None:
        self._window._reset_autosave_timer()

    def _apply_keybinds(self) -> None:
        # Pan directions route through apply_keybind() to their MapView bindings.
        for action_id, _label, _default in settings.REBINDABLE_ACTIONS:
            self._window.apply_keybind(action_id)

    def _apply_wheel_bindings(self) -> None:
        self._window.map_view.set_wheel_bindings(settings.get_wheel_bindings())
