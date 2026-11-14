"""The Settings dialog and the widgets only it uses.

Split out of viewer.py; viewer_menus.py opens it. It names ViewerWindow in annotations
only, so it imports that under TYPE_CHECKING: a runtime import back would be
a cycle."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from PyQt5.QtCore import QEvent, QSize, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontInfo,
    QFontMetrics,
    QKeySequence,
    QPalette,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from descape import (
    asset_source,
    autosave,
    content_roots,
    debug_log,
    game_dataset,
    games,
    grid_overlay,
    iso_geometry,
    mod_graphics,
    object_catalog,
    settings,
    settings_file,
    terrain_catalog,
    theme_file,
    themes,
    unit_sprites,
    view_layers,
)
from descape.tool_defaults_page import ToolDefaultsPage
from descape.viewer_common import (
    LabelFitButton,
    NoWheelComboBox,
    _sync_hi_res,
    status_color,
)
from descape.viewer_dialogs import apply_theme, apply_ui_font

if TYPE_CHECKING:
    from descape.viewer import ViewerWindow


class _VerticalScrollArea(QScrollArea):
    """Scrolls vertically only: its minimum width tracks the page's live minimum
    plus a vertical scrollbar, so a UI font bump widens the dialog, not the page (TASK-210)."""

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        page = self.widget()
        if page is None:
            return hint
        width = page.minimumSizeHint().width() + self.verticalScrollBar().sizeHint().width() + 2 * self.frameWidth()
        return QSize(max(hint.width(), width), hint.height())


class _WholeLabelComboBox(NoWheelComboBox):
    """Its minimum is its full sizeHint, re-measured on a font change, so the longest item never clips (TASK-375)."""

    def minimumSizeHint(self) -> QSize:
        # sizeHint alone leaves the edit field 2 px short of the label at 168 DPI.
        hint = self.sizeHint()
        return QSize(hint.width() + self.fontMetrics().horizontalAdvance(" "), hint.height())


class _KeyBox(QKeySequenceEdit):
    """As wide as the widest of `samples` shown whole, no wider, so the Keybinds labels
    take the slack (TASK-498). Re-measured on a font change."""

    def __init__(self, sequence: QKeySequence, samples: list[str]) -> None:
        super().__init__(sequence)
        self._samples = [QKeySequence(s).toString(QKeySequence.NativeText) for s in samples]

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        line_edit = self.findChild(QLineEdit)
        if line_edit is None:
            return hint
        metrics = line_edit.fontMetrics()
        text = max(metrics.horizontalAdvance(s) for s in self._samples)
        # QLineEdit's sizeHint is 17 'x' wide plus its frame and margins; swap the text part.
        return QSize(line_edit.sizeHint().width() - 17 * metrics.horizontalAdvance("x") + text, hint.height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class _TextWidthButton(LabelFitButton):
    """LabelFitButton whose hint is its label's width too, so the Keybinds labels get the slack (TASK-498)."""

    def sizeHint(self) -> QSize:
        return self.minimumSizeHint()


class _FontPicker(QFontComboBox):
    """Its minimum is about 12 characters, not the longest installed family, so a long
    name is cut off in the closed box; the open list still shows each name whole (TASK-498)."""

    def __init__(self) -> None:
        super().__init__()
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        # It is editable, and its line edit scrolls to the cursor: unfocused, keep a cut name's start in view.
        self.lineEdit().textChanged.connect(self._show_name_start)

    def _show_name_start(self) -> None:
        if not self.lineEdit().hasFocus():
            self._home()

    def _home(self) -> None:
        self.lineEdit().setCursorPosition(0)

    def focusOutEvent(self, event) -> None:  # Qt override; the combo is its line edit's focus proxy
        super().focusOutEvent(event)
        # Deferred: editingFinished re-sets the same text after this, putting the cursor back at the end.
        QTimer.singleShot(0, self._home)

    def showPopup(self) -> None:
        super().showPopup()
        # QFontComboBox pins the list to 5/3 of the box's width on every resize.
        view = self.view()
        popup = view.window()
        # Its delegate's width hint leaves out the icon drawn before the name, about a row high.
        need = view.sizeHintForColumn(0) + view.sizeHintForRow(0) + popup.width() - view.viewport().width()
        if need <= popup.width():
            return
        screen = self.screen().availableGeometry()
        popup.setFixedWidth(min(need, screen.width()))
        if popup.geometry().right() > screen.right():
            popup.move(screen.right() + 1 - popup.width(), popup.y())


# Extra space above and below each Settings > Appearance section rule (TASK-254).
_SECTION_GAP = 6
# The Game data tooltip counts the other choices in words.
_COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


class SettingsDialog(QDialog):
    """Tabbed settings dialog: General (app-wide behavior, including the
    AoE2DE install path -- not its own tab since it's a one-time setup
    step, not a recurring one), Mods (the profile folder and Mod art
    list), Appearance (theme, render quality), Tools (what each Tool
    Options control starts at), Saving, Keybinds and Advanced. Split into
    tabs so more settings in any category have an obvious place to land
    without reshaping this dialog."""

    def __init__(self, parent: ViewerWindow):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        # Width grows to the tabs' minimum; 640 tall still fits a 768 px screen (TASK-210).
        self.resize(520, 640)
        self._window = parent

        tabs = QTabWidget()
        # Scrolls (GH #126); see-through for the install status label.
        tabs.addTab(self._scrolled_tab(self._build_general_tab(), see_through=True), "General")
        # GH #244: the profile folder and Mod art list; see-through for their status label.
        tabs.addTab(self._scrolled_tab(self._build_mods_tab(), see_through=True), "Mods")
        # TASK-210: Appearance alone outgrows a 768 px screen, so it scrolls.
        tabs.addTab(self._scrolled_tab(self._build_appearance_tab()), "Appearance")
        # TASK-272: what each Tool Options control starts at.
        self.tool_defaults_page = ToolDefaultsPage(parent)
        tabs.addTab(self._scrolled_tab(self.tool_defaults_page), "Tools")
        tabs.addTab(self._build_saving_tab(), "Saving")
        tabs.addTab(self._build_keybinds_tab(), "Keybinds")
        tabs.addTab(self._build_advanced_tab(), "Advanced")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)
        # TASK-500: wider than 520 when the tab titles need it, so none scrolls out of view at any UI
        # font; never past the screen. The minimum stays the pages' own.
        margins = layout.contentsMargins()
        width = max(self.width(), tabs.tabBar().sizeHint().width() + margins.left() + margins.right())
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            width = min(width, max(self.minimumSizeHint().width(), screen.availableGeometry().width()))
        self.resize(width, self.height())

    @staticmethod
    def _scrolled_tab(page: QWidget, *, see_through: bool = False) -> QScrollArea:
        scroll = _VerticalScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(page)
        if see_through:
            # The tab page shows through, the background status_color(on_tab_page=True) assumes.
            # setWidget() turned the page's own fill on, so both are cleared.
            scroll.viewport().setAutoFillBackground(False)
            page.setAutoFillBackground(False)
        return scroll

    def _build_general_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        zoom_label = QLabel("Mouse wheel zoom:")
        layout.addWidget(zoom_label)
        self.zoom_cursor_radio = QRadioButton("Zoom in centered on mouse cursor location")
        self.zoom_view_center_radio = QRadioButton("Zoom in overall (view center)")
        zoom_group = QButtonGroup(tab)
        zoom_group.addButton(self.zoom_cursor_radio)
        zoom_group.addButton(self.zoom_view_center_radio)
        if settings.get_zoom_centered_on_cursor():
            self.zoom_cursor_radio.setChecked(True)
        else:
            self.zoom_view_center_radio.setChecked(True)
        # Connecting only one radio's toggled is enough: within an exclusive
        # QButtonGroup, this one flips to False exactly when the other flips
        # to True, so both directions are covered by this single signal.
        self.zoom_cursor_radio.toggled.connect(self._on_zoom_mode_toggled)
        layout.addWidget(self.zoom_cursor_radio)
        layout.addWidget(self.zoom_view_center_radio)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        self._add_install_section(layout)

        if sys.platform.startswith("linux"):
            self.xwayland_checkbox = QCheckBox("X11 compatibility on Wayland")
            self.xwayland_checkbox.setChecked(settings.get_xwayland_on_wayland())
            self.xwayland_checkbox.setToolTip(
                "On a Wayland session, run under XWayland (QT_QPA_PLATFORM=xcb) "
                "instead of natively. Try it if map drags lag or freeze for "
                "seconds at a time. Costs a blurry window under fractional "
                "display scaling. An explicit QT_QPA_PLATFORM other than "
                "wayland still wins."
            )
            self.xwayland_checkbox.toggled.connect(self._on_xwayland_toggled)
            layout.addWidget(self.xwayland_checkbox)
            # Under the checkbox text, by the indicator's own width (TASK-498).
            hint_row = QHBoxLayout()
            style = self.style()
            hint_row.addSpacing(
                style.pixelMetric(QStyle.PM_IndicatorWidth) + style.pixelMetric(QStyle.PM_CheckBoxLabelSpacing)
            )
            self.xwayland_hint = QLabel("Linux only. Takes effect next launch.")
            self.xwayland_hint.setWordWrap(True)
            hint_row.addWidget(self.xwayland_hint, stretch=1)
            layout.addLayout(hint_row)

        self._add_settings_file_section(layout)

        layout.addStretch(1)
        return tab

    def _build_mods_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        # Profile first: its refresh also refreshes the mod art list once that exists.
        self._add_profile_section(layout)
        self._add_mod_art_section(layout)
        self._add_data_mod_section(layout)
        layout.addStretch(1)
        return tab

    def _on_xwayland_toggled(self, enabled: bool) -> None:
        settings.set_xwayland_on_wayland(enabled)
        state = "on" if enabled else "off"
        self._window._log_status(f"X11 compatibility on Wayland: {state} (takes effect next launch)")

    def _on_zoom_mode_toggled(self, centered_on_cursor: bool) -> None:
        settings.set_zoom_centered_on_cursor(centered_on_cursor)
        self._window.map_view.set_zoom_anchor_mode(centered_on_cursor)
        mode = "centered on mouse cursor" if centered_on_cursor else "overall (view center)"
        self._window._log_status(f"Zoom mode: {mode}")

    def _add_settings_file_section(self, layout: QVBoxLayout) -> None:
        """GH #268: Export settings… / Import settings…, and the line for an import staged for the next launch."""
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)
        layout.addWidget(QLabel("Settings file:"))
        tooltip = (
            "A settings file carries your preferences from every tab, the View menu and the panels: keybinds, "
            "the look, tool defaults and the rest. Not folder paths, recent files, the New Map dialog's last "
            "choices, File > Data Mod's choice for each map, window layout or X11 compatibility on Wayland. "
            "An import replaces your preferences and takes effect the next time DEscape starts."
        )
        self.settings_export_button = QPushButton("Export settings…")
        self.settings_export_button.setToolTip(tooltip)
        self.settings_export_button.clicked.connect(self._export_settings)
        self.settings_import_button = QPushButton("Import settings…")
        self.settings_import_button.setToolTip(tooltip)
        self.settings_import_button.clicked.connect(self._import_settings)
        button_row = QHBoxLayout()
        button_row.addWidget(self.settings_export_button)
        button_row.addWidget(self.settings_import_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        self.settings_pending_row = QWidget()
        pending_layout = QHBoxLayout(self.settings_pending_row)
        pending_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_pending_label = QLabel()
        self.settings_pending_label.setWordWrap(True)
        pending_layout.addWidget(self.settings_pending_label, stretch=1)
        self.settings_pending_discard_button = QPushButton("Discard")
        self.settings_pending_discard_button.setToolTip("Drop the staged import; nothing changes at the next launch.")
        self.settings_pending_discard_button.clicked.connect(self._discard_pending_import)
        pending_layout.addWidget(self.settings_pending_discard_button)
        layout.addWidget(self.settings_pending_row)
        self._refresh_pending_import_row()

    def _refresh_pending_import_row(self) -> None:
        pending = settings_file.pending_import()
        self.settings_pending_row.setVisible(pending is not None)
        if pending is None:
            return
        if pending.refusal is not None:
            self.settings_pending_label.setText(f"A staged settings import will not apply: {pending.refusal}.")
            self.settings_pending_label.setToolTip(str(settings_file.pending_path()))
            return
        name = Path(pending.source).name if pending.source is not None else "an earlier import"
        self.settings_pending_label.setText(f"Settings from {name} apply the next time DEscape starts.")
        self.settings_pending_label.setToolTip(pending.source or "")

    def _export_settings(self) -> None:
        # A slider still on its apply timer must be in config.yaml before the file is built from it.
        self._flush_pending_applies()
        start = str(Path.home() / f"descape{settings_file.FILE_SUFFIX}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export settings", start, f"DEscape settings (*{settings_file.FILE_SUFFIX})"
        )
        if not path:
            return
        if not path.endswith((".yaml", ".yml")):
            path += settings_file.FILE_SUFFIX
            # The dialog's overwrite question covered the unsuffixed name only.
            if Path(path).exists():
                answer = QMessageBox.question(
                    self,
                    "Export settings",
                    f"{path} already exists. Replace it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    self._window._log_status("Settings export cancelled")
                    return
        try:
            settings_file.write_settings_file(path, settings_file.build_document())
        except settings_file.SettingsWriteBlockedError as e:
            self._window._log_status(f"Settings export blocked: {e}", level=debug_log.WARNING)
            QMessageBox.critical(self, "Export blocked", str(e))
            return
        except settings_file.ConfigUnreadable as e:
            self._window._log_status(f"Settings export failed: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Settings unreadable", f"{e}\n\nNothing was exported.")
            return
        except OSError as e:
            self._window._log_status(f"Settings export failed: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self._window._log_status(f"Settings exported to {path}")

    def _import_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import settings",
            str(Path.home()),
            f"DEscape settings (*{settings_file.FILE_SUFFIX} *.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        try:
            plan = settings_file.parse_settings_file(path, installed_families=QFontDatabase().families())
        except settings_file.SettingsFileRefused as e:
            self._window._log_status(f"Settings import refused: {e}", level=debug_log.WARNING)
            QMessageBox.warning(self, "Import refused", f"{path}\n\n{e}")
            return
        # The summary is against config.yaml, so a slider still on its apply timer is saved first.
        self._flush_pending_applies()
        try:
            summary = settings_file.format_import_summary(plan)
        except (settings_file.ConfigUnreadable, OSError) as e:
            self._window._log_status(f"Settings import refused: {e}", level=debug_log.WARNING)
            QMessageBox.warning(self, "Import refused", f"{e}\n\nAn import would replace it, so nothing was staged.")
            return
        if not self._confirm_settings_import(summary):
            self._window._log_status("Settings import cancelled")
            return
        try:
            settings_file.stage_import(plan, path)
        except OSError as e:
            self._window._log_status(f"Settings import failed: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Import failed", str(e))
            return
        self._refresh_pending_import_row()
        skipped = f", {len(plan.problems)} skipped" if plan.problems else ""
        self._window._log_status(
            f"Settings from {path} staged{skipped}: they apply the next time DEscape starts",
            level=debug_log.WARNING if plan.problems else debug_log.INFO,
        )
        if self._ask_quit_after_import():
            self._quit_for_import()

    def _confirm_settings_import(self, summary: str) -> bool:
        """Modal Apply/Cancel over the parsed file's summary. Its own method so
        offscreen tests can answer it without exec_()."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Import settings")
        body = QPlainTextEdit(summary)
        body.setReadOnly(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(body)
        layout.addWidget(buttons)
        dialog.resize(560, 360)
        return dialog.exec_() == QDialog.Accepted

    def _ask_quit_after_import(self) -> bool:
        """True for Quit DEscape now, False for Later. Its own method so offscreen tests can answer it."""
        box = QMessageBox(self)
        box.setWindowTitle("Import settings")
        box.setIcon(QMessageBox.Information)
        box.setText("The imported settings apply the next time DEscape starts.")
        box.setInformativeText(
            f"When they do, your current settings are first copied to {settings_file.BACKUP_NAME} beside config.yaml."
        )
        quit_button = box.addButton("Quit DEscape now", QMessageBox.AcceptRole)
        later_button = box.addButton("Later", QMessageBox.RejectRole)
        box.setDefaultButton(later_button)
        box.exec_()
        return box.clickedButton() is quit_button

    def _quit_for_import(self) -> None:
        # This dialog first: a close issued under the still-open modal dialog would not end the event loop.
        self.close()
        # The window's normal close, so the unsaved-changes prompt still runs.
        QTimer.singleShot(0, self._window.close)

    def _discard_pending_import(self) -> None:
        try:
            settings_file.discard_pending_import()
        except OSError as e:
            self._window._log_status(f"Discarding the staged settings import failed: {e}", level=debug_log.ERROR)
            QMessageBox.warning(self, "Discard failed", str(e))
            return
        self._refresh_pending_import_row()
        self._window._log_status("Staged settings import discarded")

    def _build_appearance_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Export and Import sit on a second line under the combo (TASK-498).
        theme_grid = QGridLayout()
        theme_grid.setColumnStretch(2, 1)
        theme_grid.addWidget(QLabel("Theme:"), 0, 0)
        self.theme_combo = QComboBox()
        for preset_id, (preset_label, _colors) in themes.PRESETS.items():
            self.theme_combo.addItem(preset_label, preset_id)
        self.theme_combo.setToolTip(
            "App chrome only (menus, dialogs, toolbars) -- the map view's own "
            "colors (real terrain textures, unit dots, hover highlights) are "
            "unaffected, since they represent game data or are already dark. "
            "The tool overlay colours below are separate and never change with it."
        )
        self.theme_combo.setCurrentIndex(self.theme_combo.findData(settings.get_theme()))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        theme_grid.addWidget(self.theme_combo, 0, 1)
        file_tooltip = (
            "A theme file carries this tab's look settings: theme and chrome colours, UI font, "
            "tool overlay colours, ruler and distance-tick label sizes, badge position, grid "
            "blend and thickness, and trigger status with its colours. Not graphics quality, "
            "preloading, elevation height, pan speed or toolbar buttons."
        )
        self.theme_export_button = QPushButton("Export theme…")
        self.theme_export_button.setToolTip(file_tooltip)
        self.theme_export_button.clicked.connect(self._export_theme)
        self.theme_import_button = QPushButton("Import theme…")
        self.theme_import_button.setToolTip(file_tooltip)
        self.theme_import_button.clicked.connect(self._import_theme)
        theme_file_row = QHBoxLayout()
        theme_file_row.addWidget(self.theme_export_button)
        theme_file_row.addWidget(self.theme_import_button)
        theme_file_row.addStretch(1)
        theme_grid.addLayout(theme_file_row, 1, 1, 1, 2)
        layout.addLayout(theme_grid)
        layout.addWidget(self._build_theme_colors_group())
        self._add_section_divider(layout)

        # Reset sits on a second line under the picker, as Export and Import do (TASK-498).
        ui_font_grid = QGridLayout()
        ui_font_grid.setColumnStretch(1, 1)
        ui_font_grid.addWidget(QLabel("UI font:"), 0, 0)
        app_font = QApplication.instance().font()
        self.ui_font_combo = _FontPicker()
        self.ui_font_size_spin = QSpinBox()
        self.ui_font_size_spin.setRange(settings.UI_FONT_SIZE_MIN, settings.UI_FONT_SIZE_MAX)
        self.ui_font_size_spin.setSuffix(" pt")
        ui_font_tooltip = (
            "App chrome only (menus, dialogs, panels, toolbars, the status "
            "log) -- the map view's own overlay text (ruler readout, "
            "distance-tick numbers, stacked-unit badges) keeps its own fixed "
            "sizes, since it is measured in device pixels against fixed label "
            "boxes. Reset returns both to the platform default."
        )
        self.ui_font_combo.setToolTip(ui_font_tooltip)
        self.ui_font_size_spin.setToolTip(ui_font_tooltip)
        ui_font_reset = QPushButton("Reset to default")
        ui_font_reset.setToolTip(ui_font_tooltip)
        # Values first, signals after -- same order theme_combo and
        # graphics_quality_slider use, and load-bearing here: populating a
        # QFontComboBox emits currentFontChanged, which connected first would
        # persist a family the user never chose and pin "" (follow the
        # platform) to whatever the platform default happened to be.
        self._set_ui_font_widgets(settings.get_ui_font_family(), settings.get_ui_font_size(), app_font)
        self._ui_font_apply_timer = QTimer(self)
        self._ui_font_apply_timer.setSingleShot(True)
        self._ui_font_apply_timer.timeout.connect(self._apply_ui_font)
        self.ui_font_combo.currentFontChanged.connect(self._on_ui_font_family_changed)
        self.ui_font_size_spin.valueChanged.connect(self._on_ui_font_size_changed)
        ui_font_reset.clicked.connect(self._on_ui_font_reset)
        ui_font_grid.addWidget(self.ui_font_combo, 0, 1)
        ui_font_grid.addWidget(self.ui_font_size_spin, 0, 2)
        ui_font_grid.addWidget(ui_font_reset, 1, 1, Qt.AlignLeft)
        layout.addLayout(ui_font_grid)

        toolbar_style_row = QHBoxLayout()
        toolbar_style_row.addWidget(QLabel("Toolbar buttons:"))
        self.toolbar_style_combo = QComboBox()
        for style_id, style_label in settings.TOOLBAR_STYLES:
            self.toolbar_style_combo.addItem(style_label, style_id)
        self.toolbar_style_combo.setCurrentIndex(self.toolbar_style_combo.findData(settings.get_toolbar_style()))
        self.toolbar_style_combo.setToolTip(
            "How the main toolbar's tool buttons show. Each tool's tooltip names it and its key either way"
        )
        self.toolbar_style_combo.currentIndexChanged.connect(self._on_toolbar_style_changed)
        toolbar_style_row.addWidget(self.toolbar_style_combo)
        toolbar_style_row.addStretch(1)
        layout.addLayout(toolbar_style_row)
        self._add_section_divider(layout)

        preload_checkbox = QCheckBox("Preload neighbouring zoom levels")
        preload_checkbox.setChecked(settings.get_preload_zoom_levels())
        preload_checkbox.setToolTip(
            "After a map opens, resolve the next zoom levels' unit sprites in "
            "idle time so the first zoom doesn't stutter, and warm a margin of "
            "chunks just outside the viewport as you pan so scrolling into new "
            "territory doesn't stutter either. The window stays interactive "
            "throughout; the cost is a few seconds of background work per "
            "open, plus ongoing idle-time work while panning, and holding "
            "extra chunks/levels in memory. A fast zoom onto a level not "
            "ready yet shows the map briefly blurred until it sharpens. "
            "Applies immediately."
        )
        preload_checkbox.toggled.connect(self._on_preload_zoom_levels_toggled)
        layout.addWidget(preload_checkbox)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Graphics quality:"))
        self.graphics_quality_slider = QSlider(Qt.Horizontal)
        self.graphics_quality_slider.setRange(settings.GRAPHICS_QUALITY_MIN, settings.GRAPHICS_QUALITY_MAX)
        self.graphics_quality_slider.setTickInterval(1)
        self.graphics_quality_slider.setTickPosition(QSlider.TicksBelow)
        self.graphics_quality_slider.setSingleStep(1)
        self.graphics_quality_slider.setPageStep(1)
        self.graphics_quality_slider.setToolTip(
            "Scales map render resolution (see render.tile_pixels_for_map) -- "
            "applies to the next render. Enhanced only doubles resolution on "
            "large maps (already-downscaled); on small maps it's a no-op, "
            "since doubling their base resolution is well past what's been "
            "measured safe for memory. With the game's free Enhanced Graphics "
            "DLC installed, Enhanced also draws units and buildings from its "
            "high-resolution art when zoomed in; without it, they use their "
            "regular art. Return of Rome scenarios use Return of Rome's own "
            "high-resolution art, which needs no DLC. Default and Enhanced "
            "also draw zoomed-out units and buildings with smooth shrinking, "
            "when the native renderer is built."
        )
        self.graphics_quality_slider.setValue(settings.get_graphics_quality())
        self.graphics_quality_label = QLabel()
        self._update_graphics_quality_label(settings.get_graphics_quality())
        self._graphics_quality_apply_timer = QTimer(self)
        self._graphics_quality_apply_timer.setSingleShot(True)
        self._graphics_quality_apply_timer.timeout.connect(self._apply_graphics_quality)
        self.graphics_quality_slider.valueChanged.connect(self._on_graphics_quality_slider_changed)
        quality_row.addWidget(self.graphics_quality_slider, stretch=1)
        quality_row.addWidget(self.graphics_quality_label)
        layout.addLayout(quality_row)
        # GH #241: whether Enhanced can draw the high-resolution unit art.
        self.x2_art_label = QLabel()
        self.x2_art_label.setWordWrap(True)
        self._update_x2_art_label()
        layout.addWidget(self.x2_art_label)

        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("Isometric elevation height:"))
        self.elev_step_slider = QSlider(Qt.Horizontal)
        # Value space is the 1-based stop index, not the pct -- setSingleStep
        # alone only governs arrow keys and the wheel, so a drag would still
        # produce arbitrary off-stop pcts via QStyle::sliderValueFromPosition.
        # Same shape as graphics_quality_slider above.
        self.elev_step_slider.setRange(1, len(settings.ELEV_STEP_PCT_STOPS))
        self.elev_step_slider.setTickInterval(1)
        self.elev_step_slider.setTickPosition(QSlider.TicksBelow)
        self.elev_step_slider.setSingleStep(1)
        self.elev_step_slider.setPageStep(1)
        self.elev_step_slider.setToolTip(
            "How tall one elevation level stands on screen in Stepped, Sloped, and Flat with "
            "Isometric View. 100% matches the game's own height. Flat with Isometric View "
            "always renders at elevation 0, so there this only shapes the diamond tiles. "
            f"The default is Tall ({iso_geometry.ELEV_STEP_DEFAULT_PCT}%). Heights above it are "
            "real headroom: the game's elevation changes are smooth ramps over several tiles, "
            "so a taller setting hides fewer tiles behind their neighbours than one sharp step that tall would."
        )
        self.elev_step_slider.setValue(settings.elev_step_index(settings.get_elev_step_pct()))
        self.elev_step_value_label = QLabel()
        self._update_elev_step_label(settings.get_elev_step_pct())
        self._elev_step_apply_timer = QTimer(self)
        self._elev_step_apply_timer.setSingleShot(True)
        self._elev_step_apply_timer.timeout.connect(self._apply_elev_step_pct)
        self.elev_step_slider.valueChanged.connect(self._on_elev_step_slider_changed)
        self.elev_step_slider.sliderReleased.connect(self._apply_elev_step_pct)
        height_row.addWidget(self.elev_step_slider, stretch=1)
        height_row.addWidget(self.elev_step_value_label)
        layout.addLayout(height_row)

        pan_row = QHBoxLayout()
        pan_row.addWidget(QLabel("Pan speed:"))
        self.pan_speed_slider = QSlider(Qt.Horizontal)
        self.pan_speed_slider.setRange(settings.PAN_SPEED_MIN, settings.PAN_SPEED_MAX)
        self.pan_speed_slider.setTickInterval(200)
        self.pan_speed_slider.setTickPosition(QSlider.TicksBelow)
        self.pan_speed_slider.setSingleStep(50)
        self.pan_speed_slider.setPageStep(100)
        self.pan_speed_slider.setToolTip(
            "How fast holding a Pan key (Settings > Keybinds) scrolls the map, "
            "in viewport pixels per second. Measured in screen pixels like the "
            "middle-drag pan, so a fixed speed covers fewer tiles the further "
            "you zoom in. Takes effect on the next hold."
        )
        self.pan_speed_slider.setValue(settings.get_pan_speed())
        self.pan_speed_value_label = QLabel()
        self._update_pan_speed_label(settings.get_pan_speed())
        # Debounced like both sliders above, for a different reason: every
        # settings setter does a full _load_config/_save_config YAML round
        # trip, so a live valueChanged would be hundreds of disk writes
        # across one drag of this 200-2000 range.
        self._pan_speed_apply_timer = QTimer(self)
        self._pan_speed_apply_timer.setSingleShot(True)
        self._pan_speed_apply_timer.timeout.connect(self._apply_pan_speed)
        self.pan_speed_slider.valueChanged.connect(self._on_pan_speed_slider_changed)
        self.pan_speed_slider.sliderReleased.connect(self._apply_pan_speed)
        pan_row.addWidget(self.pan_speed_slider, stretch=1)
        pan_row.addWidget(self.pan_speed_value_label)
        layout.addLayout(pan_row)

        wall_row = QHBoxLayout()
        wall_row.addWidget(QLabel("Wall opacity:"))
        self.wall_opacity_slider = QSlider(Qt.Horizontal)
        self.wall_opacity_slider.setRange(view_layers.WALL_OPACITY_MIN_PCT, view_layers.WALL_OPACITY_MAX_PCT)
        self.wall_opacity_slider.setTickInterval(10)
        self.wall_opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.wall_opacity_slider.setSingleStep(5)
        self.wall_opacity_slider.setPageStep(10)
        self.wall_opacity_slider.setToolTip(
            "How opaque walls and gates draw while View > Layers > Translucent "
            "Walls is on. Lower shows more of the units behind a wall line. "
            "Where two wall pieces' art overlaps, the overlap reads darker."
        )
        self.wall_opacity_slider.setValue(settings.get_wall_opacity_pct())
        self.wall_opacity_value_label = QLabel()
        self._update_wall_opacity_label(settings.get_wall_opacity_pct())
        # Debounced like the sliders above: with the layer on, each apply rebuilds every wall sprite.
        self._wall_opacity_apply_timer = QTimer(self)
        self._wall_opacity_apply_timer.setSingleShot(True)
        self._wall_opacity_apply_timer.timeout.connect(self._apply_wall_opacity)
        self.wall_opacity_slider.valueChanged.connect(self._on_wall_opacity_slider_changed)
        self.wall_opacity_slider.sliderReleased.connect(self._apply_wall_opacity)
        wall_row.addWidget(self.wall_opacity_slider, stretch=1)
        wall_row.addWidget(self.wall_opacity_value_label)
        layout.addLayout(wall_row)
        self._add_section_divider(layout)

        layout.addWidget(self._build_overlay_colors_group())
        self._add_section_divider(layout)
        layout.addWidget(self._build_trigger_status_group())
        self._add_section_divider(layout)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Ruler label size:"))
        self.ruler_label_font_spin = QSpinBox()
        self.ruler_label_font_spin.setRange(settings.RULER_LABEL_FONT_PX_MIN, settings.RULER_LABEL_FONT_PX_MAX)
        self.ruler_label_font_spin.setSuffix(" px")
        self.ruler_label_font_spin.setValue(settings.get_ruler_label_font_px())
        self.ruler_label_font_spin.valueChanged.connect(self._on_ruler_label_font_px_changed)
        font_row.addWidget(self.ruler_label_font_spin, stretch=0)
        font_row.addStretch(1)
        layout.addLayout(font_row)

        distance_tick_font_row = QHBoxLayout()
        distance_tick_font_row.addWidget(QLabel("Distance ticks label size:"))
        self.distance_tick_font_spin = QSpinBox()
        self.distance_tick_font_spin.setRange(
            settings.DISTANCE_TICK_FONT_PX_MIN, settings.DISTANCE_TICK_FONT_PX_MAX
        )
        self.distance_tick_font_spin.setSuffix(" px")
        self.distance_tick_font_spin.setValue(settings.get_distance_tick_font_px())
        self.distance_tick_font_spin.valueChanged.connect(self._on_distance_tick_font_px_changed)
        distance_tick_font_row.addWidget(self.distance_tick_font_spin, stretch=0)
        distance_tick_font_row.addStretch(1)
        layout.addLayout(distance_tick_font_row)

        # GH #100: where each stacked-unit count badge sits on its tile.
        stack_badge_row = QHBoxLayout()
        stack_badge_row.addWidget(QLabel("Stacked-unit badge position:"))
        self.stack_badge_position_combo = QComboBox()
        for position_id, position_label in settings.STACK_BADGE_POSITIONS:
            self.stack_badge_position_combo.addItem(position_label, position_id)
        self.stack_badge_position_combo.setCurrentIndex(
            self.stack_badge_position_combo.findData(settings.get_stack_badge_position())
        )
        self.stack_badge_position_combo.currentIndexChanged.connect(self._on_stack_badge_position_changed)
        stack_badge_row.addWidget(self.stack_badge_position_combo, stretch=0)
        stack_badge_row.addStretch(1)
        layout.addLayout(stack_badge_row)

        # GH #161: the Terrain-mode grid's swatch size. Not in a theme file, so not in _refresh_appearance_widgets.
        terrain_grid_row = QHBoxLayout()
        terrain_grid_row.addWidget(QLabel("Terrain grid swatch size:"))
        self.terrain_grid_swatch_combo = QComboBox()
        for px in settings.TERRAIN_GRID_SWATCH_SIZES:
            self.terrain_grid_swatch_combo.addItem(f"{px} px", px)
        self.terrain_grid_swatch_combo.setCurrentIndex(
            self.terrain_grid_swatch_combo.findData(settings.get_terrain_grid_swatch_px())
        )
        self.terrain_grid_swatch_combo.setToolTip("How big each terrain swatch is when Terrain mode's picker shows a grid.")
        self.terrain_grid_swatch_combo.currentIndexChanged.connect(self._on_terrain_grid_swatch_changed)
        terrain_grid_row.addWidget(self.terrain_grid_swatch_combo, stretch=0)
        terrain_grid_row.addStretch(1)
        layout.addLayout(terrain_grid_row)

        # View > Grid's appearance. A drag is bracketed rather than debounced:
        # its first tick swaps the baked grid for GridItem's preview (one
        # eviction), each tick after that is two QPens and an update(), and
        # sliderReleased or the timer (keyboard and wheel) re-bakes. Blend is
        # a plain value-space slider, so it moves smoothly across its range;
        # only thickness has stops left to snap to.
        self._grid_apply_timer = QTimer(self)
        self._grid_apply_timer.setSingleShot(True)
        self._grid_apply_timer.timeout.connect(self._end_grid_preview)
        grid_blend_row = QHBoxLayout()
        grid_blend_row.addWidget(QLabel("Grid line blend:"))
        self.grid_blend_slider = self._blend_slider(settings.get_grid_blend())
        self.grid_blend_value_label = QLabel(self._blend_label(settings.get_grid_blend()))
        self.grid_blend_slider.valueChanged.connect(self._on_grid_blend_changed)
        self.grid_blend_slider.sliderReleased.connect(self._end_grid_preview)
        grid_blend_row.addWidget(self.grid_blend_slider, stretch=1)
        grid_blend_row.addWidget(self.grid_blend_value_label)
        layout.addLayout(grid_blend_row)

        grid_thickness_row = QHBoxLayout()
        grid_thickness_row.addWidget(QLabel("Grid thickness:"))
        self.grid_thickness_slider = self._stop_slider(
            len(grid_overlay.THICKNESS_STOPS), grid_overlay.thickness_index(settings.get_grid_thickness())
        )
        self.grid_thickness_value_label = QLabel(f"{settings.get_grid_thickness()} px")
        self.grid_thickness_slider.valueChanged.connect(self._on_grid_thickness_changed)
        self.grid_thickness_slider.sliderReleased.connect(self._end_grid_preview)
        grid_thickness_row.addWidget(self.grid_thickness_slider, stretch=1)
        grid_thickness_row.addWidget(self.grid_thickness_value_label)
        layout.addLayout(grid_thickness_row)

        layout.addStretch(1)
        return tab

    @staticmethod
    def _add_section_divider(layout: QVBoxLayout) -> None:
        """TASK-254: a sunken rule with breathing room between Appearance's sections."""
        layout.addSpacing(_SECTION_GAP)
        divider = QFrame()
        divider.setObjectName("appearance_section_divider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)
        layout.addSpacing(_SECTION_GAP)

    @staticmethod
    def _stop_slider(stop_count: int, index: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, stop_count)
        slider.setTickInterval(1)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setValue(index)
        return slider

    @staticmethod
    def _blend_slider(value: int) -> QSlider:
        """Every value in the range, not a stop ladder. The ticks are only a
        landmark for the invisible centre; dragging still lands anywhere."""
        slider = QSlider(Qt.Horizontal)
        slider.setRange(grid_overlay.BLEND_MIN, grid_overlay.BLEND_MAX)
        slider.setTickInterval(grid_overlay.BLEND_MAX // 2)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setValue(value)
        return slider

    @staticmethod
    def _blend_label(value: int) -> str:
        if not value:
            return "off"
        return f"{'light' if value > 0 else 'dark'} {abs(value)}"

    def _on_grid_blend_changed(self, value: int) -> None:
        settings.set_grid_blend(value)
        self.grid_blend_value_label.setText(self._blend_label(value))
        self._preview_grid_appearance()

    def _on_grid_thickness_changed(self, index: int) -> None:
        value = grid_overlay.thickness_for_index(index)
        settings.set_grid_thickness(value)
        self.grid_thickness_value_label.setText(f"{value} px")
        self._preview_grid_appearance()

    def _preview_grid_appearance(self) -> None:
        window = self._window
        view = window.map_view
        if view.grid_bake_live() and not view.grid_previewing():
            window._apply_grid_change(view.begin_grid_preview)
        view.set_grid_appearance(settings.get_grid_blend(), settings.get_grid_thickness())
        if view.grid_previewing() and not (
            self.grid_blend_slider.isSliderDown() or self.grid_thickness_slider.isSliderDown()
        ):
            self._grid_apply_timer.start(200)

    def _end_grid_preview(self) -> None:
        self._grid_apply_timer.stop()
        if self._window.map_view.grid_previewing():
            self._window._apply_grid_change(self._window.map_view.end_grid_preview)

    def done(self, result: int) -> None:
        self._flush_pending_applies()
        super().done(result)

    def closeEvent(self, event) -> None:
        # close() on a never-shown dialog skips done(); its timers must not outlive it either.
        self._flush_pending_applies()
        super().closeEvent(event)

    def _flush_pending_applies(self) -> None:
        # A keyboard change's timer dies with the dialog, which would leave
        # the grid stuck in its unbaked preview.
        self._end_grid_preview()
        # Same for a pending opacity: its override would stay live but never persist.
        if self._overlay_opacity_timer.isActive():
            self._persist_overlay_opacity()
        # Applied now, not on a timer that can fire after the window (or a test's settings isolation) is gone.
        for timer, apply in (
            (self._ui_font_apply_timer, self._apply_ui_font),
            (self._graphics_quality_apply_timer, self._apply_graphics_quality),
            (self._elev_step_apply_timer, self._apply_elev_step_pct),
            (self._pan_speed_apply_timer, self._apply_pan_speed),
            (self._wall_opacity_apply_timer, self._apply_wall_opacity),
        ):
            if timer.isActive():
                timer.stop()
                apply()

    # OVERLAY_COLORS id prefix (before the first "_") -> section header text,
    # mirroring settings.KEYBIND_SECTION_TITLES.
    _OVERLAY_SECTION_TITLES: ClassVar[dict[str, str]] = {
        "highlight": "Terrain brush",
        "pan": "Pan",
        "unit": "Units",
        "ruler": "Ruler",
        "region": "Select tool",
        "mirror": "Map mirroring",
        "footprint": "Footprint outlines",
        "fish": "Fish outlines",
        "range": "Range rings",
        "analysis": "Map Analysis",
        "trigger": "Trigger overlay",
    }
    # Colour section prefix -> settings.OVERLAY_OPACITIES group; its slider closes that section.
    _OVERLAY_OPACITY_SECTIONS: ClassVar[dict[str, str]] = {
        "unit": "unit_select",
        "region": "region",
        "trigger": "trigger_area",
    }

    def _add_overlay_opacity_row(self, grid: QGridLayout, row: int, section: str) -> int:
        """The section's opacity row, if it has one, in the swatch rows' 3-column grid."""
        group_id = self._OVERLAY_OPACITY_SECTIONS.get(section)
        if group_id is None:
            return row
        label = next(text for gid, text, _default in settings.OVERLAY_OPACITIES if gid == group_id)
        grid.addWidget(QLabel(label), row, 0)

        slider = QSlider(Qt.Horizontal)
        slider.setObjectName(f"overlay_opacity_{group_id}")
        slider.setRange(settings.OVERLAY_OPACITY_MIN, settings.OVERLAY_OPACITY_MAX)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setToolTip(
            "Fades the whole group (fill and outlines) on top of its colours. 100% is the original look."
        )
        # Set before connecting, or building the dialog would push a change.
        slider.setValue(settings.get_overlay_opacity(group_id))
        value_label = QLabel(f"{slider.value()}%")
        value_label.setMinimumWidth(QFontMetrics(value_label.font()).horizontalAdvance("100%"))
        slider.valueChanged.connect(lambda value, gid=group_id: self._on_overlay_opacity_changed(gid, value))
        slider.sliderReleased.connect(self._persist_overlay_opacity)
        cell = QHBoxLayout()
        cell.setContentsMargins(0, 0, 0, 0)
        cell.addWidget(slider, stretch=1)
        cell.addWidget(value_label)
        grid.addLayout(cell, row, 1)

        default_btn = QPushButton("Default")
        default_btn.clicked.connect(lambda _checked, gid=group_id: self._reset_overlay_opacity(gid))
        grid.addWidget(default_btn, row, 2)

        self._overlay_opacity_sliders[group_id] = slider
        self._overlay_opacity_labels[group_id] = value_label
        self._overlay_opacity_defaults[group_id] = default_btn
        return row + 1

    def _pending_overlay_opacity(self) -> dict[str, int]:
        return {
            gid: slider.value()
            for gid, slider in self._overlay_opacity_sliders.items()
            if slider.value() != settings.get_overlay_opacity(gid)
        }

    def _on_overlay_opacity_changed(self, group_id: str, value: int) -> None:
        self._overlay_opacity_labels[group_id].setText(f"{value}%")
        # Live but unpersisted, so a drag fades the map without a disk write per step.
        self._window.map_view.apply_overlay_opacity(overrides=self._pending_overlay_opacity())
        if not self._overlay_opacity_sliders[group_id].isSliderDown():
            self._overlay_opacity_timer.start(200)

    def _persist_overlay_opacity(self) -> None:
        self._overlay_opacity_timer.stop()
        pending = self._pending_overlay_opacity()
        for group_id, pct in pending.items():
            settings.set_overlay_opacity(group_id, pct)
        self._window.map_view.apply_overlay_opacity()
        for group_id, pct in pending.items():
            section = next(s for s, gid in self._OVERLAY_OPACITY_SECTIONS.items() if gid == group_id)
            self._window._log_status(f"Overlay opacity: {self._OVERLAY_SECTION_TITLES[section]} -> {pct}%")

    def _reset_overlay_opacity(self, group_id: str) -> None:
        self._overlay_opacity_sliders[group_id].setValue(settings.get_default_overlay_opacity(group_id))
        self._persist_overlay_opacity()

    def _build_overlay_colors_group(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(QLabel("Tool overlay colors:"))

        rows_widget = QWidget()
        grid = QGridLayout(rows_widget)
        grid.setColumnStretch(1, 1)

        self._overlay_swatches: dict[str, QPushButton] = {}
        # GH #129 opacity rows: kept apart from _overlay_swatches, which maps colour ids only.
        self._overlay_opacity_sliders: dict[str, QSlider] = {}
        self._overlay_opacity_labels: dict[str, QLabel] = {}
        self._overlay_opacity_defaults: dict[str, QPushButton] = {}
        # One shared debounce, for pan_speed_slider's reason: every setter is a full YAML round trip.
        self._overlay_opacity_timer = QTimer(self)
        self._overlay_opacity_timer.setSingleShot(True)
        self._overlay_opacity_timer.timeout.connect(self._persist_overlay_opacity)
        row = 0
        current_section = None
        for color_id, label, _default in settings.OVERLAY_COLORS:
            section = color_id.split("_", 1)[0]
            if section != current_section:
                if current_section is not None:
                    row = self._add_overlay_opacity_row(grid, row, current_section)
                    divider = QFrame()
                    divider.setFrameShape(QFrame.HLine)
                    divider.setFrameShadow(QFrame.Sunken)
                    grid.addWidget(divider, row, 0, 1, 3)
                    row += 1
                section_label = QLabel(f"<b>{self._OVERLAY_SECTION_TITLES.get(section, section.title())}</b>")
                grid.addWidget(section_label, row, 0, 1, 3)
                row += 1
                current_section = section

            self._overlay_swatches[color_id] = self._add_swatch_row(
                grid,
                row,
                label,
                settings.get_overlay_color(color_id),
                lambda cid=color_id: self._pick_overlay_color(cid),
                lambda cid=color_id: self._reset_overlay_color(cid),
            )
            row += 1

        if current_section is not None:
            row = self._add_overlay_opacity_row(grid, row, current_section)
        grid.setRowStretch(row, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_widget)
        scroll.setMinimumHeight(220)
        outer.addWidget(scroll)
        return container

    def _add_swatch_row(
        self,
        grid: QGridLayout,
        row: int,
        label: str,
        hex_str: str,
        on_pick: Callable[[], None],
        on_default: Callable[[], None],
    ) -> QPushButton:
        """One label / colour swatch / Default row in a 3-column grid, shared
        by the tool overlay, chrome colour and trigger status groups."""
        grid.addWidget(QLabel(label), row, 0)
        swatch = QPushButton()
        swatch.setFixedWidth(60)
        swatch.clicked.connect(lambda _checked: on_pick())
        self._set_swatch_color(swatch, hex_str)
        # Left-aligned so column 1, not Default, takes the grid's spare width.
        grid.addWidget(swatch, row, 1, Qt.AlignLeft)
        default_btn = QPushButton("Default")
        default_btn.clicked.connect(lambda _checked: on_default())
        grid.addWidget(default_btn, row, 2)
        return swatch

    def _build_theme_colors_group(self) -> QWidget:
        """Per-role chrome colour overrides on top of the selected preset.
        Each swatch shows the live palette's colour, so an unset role shows
        the preset's; Default drops that role's override."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel("Chrome colours:"))
        header.addStretch(1)
        self.theme_reset_button = QPushButton("Reset all to preset")
        self.theme_reset_button.clicked.connect(self._reset_theme_colors)
        header.addWidget(self.theme_reset_button)
        outer.addLayout(header)

        rows_widget = QWidget()
        grid = QGridLayout(rows_widget)
        grid.setColumnStretch(1, 1)
        self._theme_swatches: dict[str, QPushButton] = {}
        for row, (role_id, label, _qt_name) in enumerate(themes.THEME_ROLES):
            self._theme_swatches[role_id] = self._add_swatch_row(
                grid,
                row,
                label,
                self._live_theme_color(role_id),
                lambda rid=role_id: self._pick_theme_color(rid),
                lambda rid=role_id: self._reset_theme_color(rid),
            )
        grid.setRowStretch(len(themes.THEME_ROLES), 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_widget)
        scroll.setMinimumHeight(140)
        outer.addWidget(scroll)
        return container

    @staticmethod
    def _live_theme_color(role_id: str) -> str:
        role = getattr(QPalette, themes.ROLE_QT_NAMES[role_id])
        # The Button pair lives in the push-button palette; the app palette's Button is the surface.
        palette = QApplication.palette("QPushButton") if role_id in themes.PUSH_BUTTON_ROLES else QApplication.palette()
        return palette.color(QPalette.Active, role).name()

    def _refresh_theme_swatches(self) -> None:
        for role_id, swatch in self._theme_swatches.items():
            self._set_swatch_color(swatch, self._live_theme_color(role_id))

    def _apply_chrome_theme(self) -> None:
        apply_theme(QApplication.instance(), settings.get_theme(), settings.get_theme_colors())
        self._refresh_theme_swatches()

    # Each status label's last `ok`, None until it is first set.
    _install_status_ok: bool | None = None
    _profile_status_ok: bool | None = None
    _autosave_dir_status_ok: bool | None = None

    def changeEvent(self, event) -> None:
        # A theme change moves the tab page under the status labels, so their colours are re-derived.
        if event.type() == QEvent.PaletteChange:
            for ok, label in (
                (self._install_status_ok, getattr(self, "install_status_label", None)),
                (self._profile_status_ok, getattr(self, "profile_status_label", None)),
                (self._autosave_dir_status_ok, getattr(self, "autosave_dir_status_label", None)),
            ):
                if ok is not None:
                    label.setStyleSheet(f"color: {status_color(ok, on_tab_page=True)};")
        super().changeEvent(event)

    def _on_theme_changed(self, index: int) -> None:
        preset_id = self.theme_combo.itemData(index)
        if preset_id == settings.get_theme():
            return
        overrides = settings.get_theme_colors()
        if overrides:
            answer = QMessageBox.question(
                self,
                "Switch theme",
                f"Switching preset clears your {len(overrides)} custom chrome colour(s). Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self.theme_combo.blockSignals(True)
                self.theme_combo.setCurrentIndex(self.theme_combo.findData(settings.get_theme()))
                self.theme_combo.blockSignals(False)
                return
        settings.set_theme(preset_id)
        self._apply_chrome_theme()
        self._window._log_status(f"Theme: {themes.preset_label(preset_id)}")

    def _pick_theme_color(self, role_id: str) -> None:
        chosen = QColorDialog.getColor(QColor(self._live_theme_color(role_id)), self)
        if not chosen.isValid():
            return
        settings.set_theme_color(role_id, chosen.name())
        self._apply_chrome_theme()
        self._window._log_status(f"Chrome colour: {themes.role_label(role_id)} -> {chosen.name()}")

    def _reset_theme_color(self, role_id: str) -> None:
        settings.clear_theme_color(role_id)
        self._apply_chrome_theme()
        self._window._log_status(f"Chrome colour: {themes.role_label(role_id)} -> preset")

    def _reset_theme_colors(self) -> None:
        # set_theme() on the current preset is the one-save "clear every override".
        settings.set_theme(settings.get_theme())
        self._apply_chrome_theme()
        self._window._log_status(f"Chrome colours: reset to {themes.preset_label(settings.get_theme())}")

    def _export_theme(self) -> None:
        start = str(Path.home() / f"descape{theme_file.FILE_SUFFIX}")
        path, _ = QFileDialog.getSaveFileName(self, "Export theme", start, f"DEscape theme (*{theme_file.FILE_SUFFIX})")
        if not path:
            return
        if not path.endswith((".yaml", ".yml")):
            path += theme_file.FILE_SUFFIX
            # The dialog's overwrite question covered the unsuffixed name only.
            if Path(path).exists():
                answer = QMessageBox.question(
                    self,
                    "Export theme",
                    f"{path} already exists. Replace it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    self._window._log_status("Theme export cancelled")
                    return
        try:
            theme_file.write_theme_file(path, theme_file.build_document())
        except theme_file.ThemeWriteBlockedError as e:
            self._window._log_status(f"Theme export blocked: {e}", level=debug_log.WARNING)
            QMessageBox.critical(self, "Export blocked", str(e))
            return
        except OSError as e:
            self._window._log_status(f"Theme export failed: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self._window._log_status(f"Theme exported to {path}")

    def _import_theme(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import theme", str(Path.home()), f"DEscape theme (*{theme_file.FILE_SUFFIX} *.yaml *.yml);;All files (*)"
        )
        if not path:
            return
        try:
            plan = theme_file.parse_theme_file(path, installed_families=QFontDatabase().families())
        except theme_file.ThemeFileRefused as e:
            self._window._log_status(f"Theme import refused: {e}", level=debug_log.WARNING)
            QMessageBox.warning(self, "Import refused", f"{path}\n\n{e}")
            return
        if not self._confirm_theme_import(theme_file.format_import_summary(plan)):
            self._window._log_status("Theme import cancelled")
            return
        self._apply_imported_appearance(plan.values)
        skipped = f", {len(plan.problems)} skipped" if plan.problems else ""
        self._window._log_status(f"Theme imported from {path}{skipped}", level=debug_log.WARNING if plan.problems else debug_log.INFO)

    def _confirm_theme_import(self, summary: str) -> bool:
        """Modal Apply/Cancel over the parsed file's summary. Its own method so
        offscreen tests can answer it without exec_()."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Import theme")
        body = QPlainTextEdit(summary)
        body.setReadOnly(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(body)
        layout.addWidget(buttons)
        dialog.resize(560, 360)
        return dialog.exec_() == QDialog.Accepted

    def _apply_imported_appearance(self, values: dict) -> None:
        """Persist in one save, refresh every Appearance widget with its signals
        blocked (so nothing re-persists), then run each live applier."""
        settings.apply_appearance_values(values)
        app = QApplication.instance()
        view = self._window.map_view
        self._end_grid_preview()
        self._refresh_appearance_widgets()
        self._apply_chrome_theme()
        apply_ui_font(app, settings.get_ui_font_family(), settings.get_ui_font_size())
        self._set_ui_font_widgets(settings.get_ui_font_family(), settings.get_ui_font_size(), app.font())
        self._window._update_log_min_height()
        view.apply_overlay_colors()
        view.apply_ruler_label_font()
        view.apply_distance_tick_font()
        view.set_stack_badge_position(settings.get_stack_badge_position())
        blend, thickness = settings.get_grid_blend(), settings.get_grid_thickness()
        self._window._apply_grid_change(lambda: view.set_grid_appearance(blend, thickness))
        self._window.trigger_panel.apply_status_style()

    def _refresh_appearance_widgets(self) -> None:
        widgets = (
            self.theme_combo,
            self.ruler_label_font_spin,
            self.distance_tick_font_spin,
            self.stack_badge_position_combo,
            self.grid_blend_slider,
            self.grid_thickness_slider,
            self.trigger_status_marker_combo,
            self.trigger_status_ok_rows_check,
            self.trigger_status_sections_check,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.theme_combo.setCurrentIndex(self.theme_combo.findData(settings.get_theme()))
            self.ruler_label_font_spin.setValue(settings.get_ruler_label_font_px())
            self.distance_tick_font_spin.setValue(settings.get_distance_tick_font_px())
            self.stack_badge_position_combo.setCurrentIndex(
                self.stack_badge_position_combo.findData(settings.get_stack_badge_position())
            )
            self.grid_blend_slider.setValue(settings.get_grid_blend())
            self.grid_thickness_slider.setValue(grid_overlay.thickness_index(settings.get_grid_thickness()))
            self.trigger_status_marker_combo.setCurrentIndex(
                self.trigger_status_marker_combo.findData(settings.get_trigger_status_marker())
            )
            self.trigger_status_ok_rows_check.setChecked(settings.get_trigger_status_color_ok_rows())
            self.trigger_status_sections_check.setChecked(settings.get_trigger_status_color_sections())
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self.grid_blend_value_label.setText(self._blend_label(settings.get_grid_blend()))
        self.grid_thickness_value_label.setText(f"{settings.get_grid_thickness()} px")
        for color_id, swatch in self._overlay_swatches.items():
            self._set_swatch_color(swatch, settings.get_overlay_color(color_id))
        for status_id, swatch in self._trigger_status_swatches.items():
            self._set_swatch_color(swatch, settings.get_trigger_status_color(status_id))

    @staticmethod
    def _set_swatch_color(swatch: QPushButton, hex_str: str) -> None:
        swatch.setStyleSheet(f"background-color: {hex_str};")
        swatch.setToolTip(hex_str)

    def _pick_overlay_color(self, color_id: str) -> None:
        current = QColor(settings.get_overlay_color(color_id))
        chosen = QColorDialog.getColor(current, self)
        if not chosen.isValid():
            return
        self._apply_overlay_color(color_id, chosen.name())

    def _reset_overlay_color(self, color_id: str) -> None:
        self._apply_overlay_color(color_id, settings.get_default_overlay_color(color_id))

    def _apply_overlay_color(self, color_id: str, hex_str: str) -> None:
        settings.set_overlay_color(color_id, hex_str)
        self._set_swatch_color(self._overlay_swatches[color_id], hex_str)
        self._window.map_view.apply_overlay_colors()
        label = settings.get_overlay_color_label(color_id)
        self._window._log_status(f"Overlay colour: {label} -> {hex_str}")

    _TRIGGER_STATUS_TIP = (
        "How the Triggers panel marks each trigger, condition and effect: complete when "
        "every required field is set and every unit or trigger it names exists, otherwise "
        "missing something (hover a marked row for what). A completeness check, not a "
        "promise the trigger works in game. The icons are a tick and a cross, so they "
        "read without colour, in their own column beside the trigger names; a section "
        "with no conditions or effects shows a dash."
    )

    def _build_trigger_status_group(self) -> QWidget:
        """GH #166's marker mode, the pass toggle and the two colours."""
        # The marker, its two toggles on their own line, then one row per colour (TASK-498).
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)

        marker_row = QHBoxLayout()
        heading = QLabel("Trigger status:")
        heading.setToolTip(self._TRIGGER_STATUS_TIP)
        marker_row.addWidget(heading)
        self.trigger_status_marker_combo = QComboBox()
        for marker_id, marker_label in settings.TRIGGER_STATUS_MARKERS:
            self.trigger_status_marker_combo.addItem(marker_label, marker_id)
        self.trigger_status_marker_combo.setCurrentIndex(
            self.trigger_status_marker_combo.findData(settings.get_trigger_status_marker())
        )
        self.trigger_status_marker_combo.setToolTip(self._TRIGGER_STATUS_TIP)
        self.trigger_status_marker_combo.currentIndexChanged.connect(self._on_trigger_status_marker_changed)
        marker_row.addWidget(self.trigger_status_marker_combo)
        marker_row.addStretch(1)
        grid.addLayout(marker_row, 0, 0, 1, 3)

        toggles_row = QHBoxLayout()
        self.trigger_status_ok_rows_check = QCheckBox("Mark triggers that pass")
        self.trigger_status_ok_rows_check.setToolTip(
            "Off leaves only the rows that are missing something marked: rows that pass get no colour and no tick"
        )
        self.trigger_status_ok_rows_check.setChecked(settings.get_trigger_status_color_ok_rows())
        self.trigger_status_ok_rows_check.toggled.connect(self._on_trigger_status_ok_rows_toggled)
        toggles_row.addWidget(self.trigger_status_ok_rows_check)
        self.trigger_status_sections_check = QCheckBox("Colour section rows")
        self.trigger_status_sections_check.setToolTip(
            "Colour a section's own row by its conditions and effects; its icon shows either way"
        )
        self.trigger_status_sections_check.setChecked(settings.get_trigger_status_color_sections())
        self.trigger_status_sections_check.toggled.connect(self._on_trigger_status_sections_toggled)
        toggles_row.addWidget(self.trigger_status_sections_check)
        toggles_row.addStretch(1)
        grid.addLayout(toggles_row, 1, 0, 1, 3)

        self._trigger_status_swatches: dict[str, QPushButton] = {}
        for row, (status_id, label, _default) in enumerate(settings.TRIGGER_STATUS_COLORS, start=2):
            self._trigger_status_swatches[status_id] = self._add_swatch_row(
                grid,
                row,
                label,
                settings.get_trigger_status_color(status_id),
                lambda sid=status_id: self._pick_trigger_status_color(sid),
                lambda sid=status_id: self._apply_trigger_status_color(
                    sid, settings.get_default_trigger_status_color(sid)
                ),
            )
        return container

    def _on_toolbar_style_changed(self, index: int) -> None:
        settings.set_toolbar_style(self.toolbar_style_combo.itemData(index))
        self._window._apply_toolbar_style()
        self._window._log_status(f"Toolbar buttons: {self.toolbar_style_combo.itemText(index)}")

    def _on_trigger_status_marker_changed(self, index: int) -> None:
        settings.set_trigger_status_marker(self.trigger_status_marker_combo.itemData(index))
        self._window.trigger_panel.apply_status_style()
        self._window._log_status(f"Trigger status marker: {self.trigger_status_marker_combo.itemText(index)}")

    def _on_trigger_status_ok_rows_toggled(self, checked: bool) -> None:
        settings.set_trigger_status_color_ok_rows(checked)
        self._window.trigger_panel.apply_status_style()
        self._window._log_status(f"Mark triggers that pass: {'on' if checked else 'off'}")

    def _on_trigger_status_sections_toggled(self, checked: bool) -> None:
        settings.set_trigger_status_color_sections(checked)
        self._window.trigger_panel.apply_status_style()
        self._window._log_status(f"Colour section rows: {'on' if checked else 'off'}")

    def _pick_trigger_status_color(self, status_id: str) -> None:
        chosen = QColorDialog.getColor(QColor(settings.get_trigger_status_color(status_id)), self)
        if chosen.isValid():
            self._apply_trigger_status_color(status_id, chosen.name())

    def _apply_trigger_status_color(self, status_id: str, hex_str: str) -> None:
        settings.set_trigger_status_color(status_id, hex_str)
        self._set_swatch_color(self._trigger_status_swatches[status_id], hex_str)
        self._window.trigger_panel.apply_status_style()
        label = settings.get_trigger_status_color_label(status_id)
        self._window._log_status(f"Trigger status colour: {label} -> {hex_str}")

    def _on_ruler_label_font_px_changed(self, value: int) -> None:
        settings.set_ruler_label_font_px(value)
        self._window.map_view.apply_ruler_label_font()
        self._window._log_status(f"Ruler label size: {value}px")

    def _on_distance_tick_font_px_changed(self, value: int) -> None:
        settings.set_distance_tick_font_px(value)
        self._window.map_view.apply_distance_tick_font()
        self._window._log_status(f"Distance ticks label size: {value}px")

    def _on_stack_badge_position_changed(self, index: int) -> None:
        position_id = self.stack_badge_position_combo.itemData(index)
        settings.set_stack_badge_position(position_id)
        self._window.map_view.set_stack_badge_position(position_id)
        self._window._log_status(f"Stacked-unit badge position: {self.stack_badge_position_combo.itemText(index)}")

    def _on_terrain_grid_swatch_changed(self, index: int) -> None:
        px = self.terrain_grid_swatch_combo.itemData(index)
        settings.set_terrain_grid_swatch_px(px)
        self._window.terrain_panel.set_grid_swatch_px(px)
        self._window._log_status(f"Terrain grid swatch size: {px} px")

    def _set_ui_font_widgets(self, family: str, size: int | None, app_font) -> None:
        """Opens the pair on the persisted values, falling back to the live
        app font for either one that is unset. Signals are blocked: this
        also runs from Reset, where re-firing them would persist exactly
        what Reset just cleared. QFontInfo, not app_font.pointSize(), since
        the latter is -1 for a pixel-sized font and QSpinBox would silently
        clamp that to the range minimum."""
        self.ui_font_combo.blockSignals(True)
        self.ui_font_size_spin.blockSignals(True)
        self.ui_font_combo.setCurrentFont(QFont(family) if family else app_font)
        current_pt = QFontInfo(app_font).pointSize()
        self.ui_font_size_spin.setValue(
            size
            if size is not None
            else max(settings.UI_FONT_SIZE_MIN, min(settings.UI_FONT_SIZE_MAX, current_pt))
        )
        self.ui_font_combo.blockSignals(False)
        self.ui_font_size_spin.blockSignals(False)

    def _on_ui_font_family_changed(self, font) -> None:
        # Persisted here, per widget, rather than in the debounced applier:
        # an applier reading both widgets would write the combo's displayed
        # family on a size-only change, turning "follow the platform" into a
        # permanent pin on whatever was showing.
        settings.set_ui_font_family(font.family())
        self._ui_font_apply_timer.start(200)

    def _on_ui_font_size_changed(self, value: int) -> None:
        settings.set_ui_font_size(value)
        self._ui_font_apply_timer.start(200)

    def _on_ui_font_reset(self) -> None:
        settings.set_ui_font_family("")
        settings.set_ui_font_size(None)
        apply_ui_font(QApplication.instance(), "", None)
        self._set_ui_font_widgets("", None, QApplication.instance().font())
        self._window._update_log_min_height()
        self._window._log_status("UI font: platform default")

    def _apply_ui_font(self) -> None:
        """Debounced, the same 200ms shape as _apply_graphics_quality and for
        the same reason: a spinbox held on its arrow fires once per step, and
        every app.setFont() is a full-app relayout."""
        family = settings.get_ui_font_family()
        size = settings.get_ui_font_size()
        apply_ui_font(QApplication.instance(), family, size)
        self._window._update_log_min_height()
        self._window._apply_left_pane_floor()  # Units, Terrain and Triggers floors are font-measured
        # Again once the posted relayout has run: until then a panel's minimumSizeHint is the old font's.
        QTimer.singleShot(0, self._window._apply_left_pane_floor)
        shown_family = family or "platform default"
        shown_size = f"{size} pt" if size is not None else "default size"
        self._window._log_status(f"UI font: {shown_family}, {shown_size}")

    def _on_preload_zoom_levels_toggled(self, enabled: bool) -> None:
        """Persists only -- turning it OFF also cancels every warm already in
        flight through _cancel_warms() (the level warm, the margin-ring warm,
        and the load-time warm with its queue), so the choice takes effect
        immediately rather than once queued work happens to finish. Turning
        it ON does not start either retroactively: the level warm's
        neighbours are derived from the fit baseline at open time (and the
        current view may be nowhere near it), and the margin warm only ever
        starts from the next viewport-changed poll fire.

        Placeholder paints follow it (TASK-031.25 D7): off also settles a
        pending level, since nothing would warm it any more."""
        settings.set_preload_zoom_levels(enabled)
        if not enabled:
            self._window._cancel_warms()
        self._window.sync_canvas_placeholders()
        self._window._log_status(f"Preload zoom levels: {'on' if enabled else 'off'}")

    def _update_x2_art_label(self) -> None:
        detected = False
        if asset_source.get_install_path() is None:
            text = "Enhanced Graphics DLC: no install configured."
        elif unit_sprites.x2_art_installed():
            detected = True
            text = "Enhanced Graphics DLC: detected. Enhanced draws its high-resolution unit art when zoomed in."
        else:
            text = "Enhanced Graphics DLC: not detected. Enhanced uses the regular unit art."
        for game in games.GAMES:
            if unit_sprites.own_x2_art_installed(game):
                name = game.display_name
                text += f" {name} scenarios draw {name}'s own high-resolution art either way."
        # Green when detected; a missing DLC is a normal setup, not an error, so it stays plain.
        self.x2_art_label.setStyleSheet(f"color: {status_color(True, on_tab_page=True)};" if detected else "")
        self.x2_art_label.setText(text)

    def _update_graphics_quality_label(self, quality: int) -> None:
        self.graphics_quality_label.setText(settings.GRAPHICS_QUALITY_LABELS[quality])

    def _on_graphics_quality_slider_changed(self, value: int) -> None:
        self._update_graphics_quality_label(value)
        # Debounced (not applied on every intermediate value) since a full
        # re-render is expensive and dragging the slider across all 4 stops
        # would otherwise fire one per stop -- see the elev_step slider's
        # matching comment below, same reasoning.
        self._graphics_quality_apply_timer.start(200)

    def _apply_graphics_quality(self) -> None:
        value = self.graphics_quality_slider.value()
        settings.set_graphics_quality(value)
        self._window._log_status(f"Graphics quality: {settings.GRAPHICS_QUALITY_LABELS[value]}{_sync_hi_res()}")
        self._window.refresh_map()

    def _update_elev_step_label(self, pct: int) -> None:
        suffix = " (Tall, default)" if pct == iso_geometry.ELEV_STEP_DEFAULT_PCT else ""
        self.elev_step_value_label.setText(f"{pct}%{suffix}")

    def _on_elev_step_slider_changed(self, value: int) -> None:
        # value is a stop index -- _update_elev_step_label still takes a pct,
        # so its "(Tall, default)" test against ELEV_STEP_DEFAULT_PCT holds.
        self._update_elev_step_label(settings.elev_step_pct_for_index(value))
        # Each application is a full Stepped-mode re-render
        # (elevations_and_proj + IsoChunkCache rebuild) -- expensive enough
        # on a real map that applying it live per-pixel would make dragging
        # itself laggy. While the mouse is down, just update the label and
        # wait for sliderReleased to apply. Keyboard arrow presses have no
        # press/release pair, so fall back to a short debounce timer for
        # those instead.
        if self.elev_step_slider.isSliderDown():
            return
        self._elev_step_apply_timer.start(200)

    def _apply_elev_step_pct(self) -> None:
        value = settings.elev_step_pct_for_index(self.elev_step_slider.value())
        settings.set_elev_step_pct(value)
        self._window._log_status(f"Isometric elevation height: {value}%")
        self._window.refresh_map()

    def _update_pan_speed_label(self, px_per_s: int) -> None:
        suffix = " (default)" if px_per_s == settings.PAN_SPEED_DEFAULT else ""
        self.pan_speed_value_label.setText(f"{px_per_s} px/s{suffix}")

    def _on_pan_speed_slider_changed(self, value: int) -> None:
        self._update_pan_speed_label(value)
        if self.pan_speed_slider.isSliderDown():
            return
        self._pan_speed_apply_timer.start(200)

    def _apply_pan_speed(self) -> None:
        # No refresh_map(): MapView._pan_step reads get_pan_speed() on every
        # tick, so the new value is live on the next hold with no wiring.
        settings.set_pan_speed(self.pan_speed_slider.value())

    def _update_wall_opacity_label(self, pct: int) -> None:
        suffix = " (default)" if pct == view_layers.WALL_OPACITY_DEFAULT_PCT else ""
        self.wall_opacity_value_label.setText(f"{pct}%{suffix}")

    def _on_wall_opacity_slider_changed(self, value: int) -> None:
        self._update_wall_opacity_label(value)
        if self.wall_opacity_slider.isSliderDown():
            return
        self._wall_opacity_apply_timer.start(200)

    def _apply_wall_opacity(self) -> None:
        value = self.wall_opacity_slider.value()
        settings.set_wall_opacity_pct(value)
        # _apply_layers() disables the window, and this dialog with it, which drops the slider's focus.
        had_focus = self.wall_opacity_slider.hasFocus()
        self._window.set_wall_opacity_pct(value)
        if had_focus:
            self.wall_opacity_slider.setFocus()

    def _build_advanced_tab(self) -> QWidget:
        """Game data (GH #67): which game draws every opened scenario. A setting
        rather than a View toggle, since anything but Auto shows the other
        game's file read-only. Set before connecting, like the Saving tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        row.addWidget(QLabel("Game data:"))
        self.game_data_combo = QComboBox()
        for choice_id, label in settings.game_data_choices():
            self.game_data_combo.addItem(label, choice_id)
        self.game_data_combo.setCurrentIndex(self.game_data_combo.findData(settings.get_game_data()))
        # One choice per registered game besides Auto, so the counts follow the registry.
        count = len(games.GAMES)
        another = "the other game" if count == 2 else "another game"
        self.game_data_combo.setToolTip(
            "Which game's units, names and art draw a scenario. Auto uses the game the "
            f"file was made for. The other {_COUNT_WORDS.get(count, count)} show every file in that game's data, and a "
            f"file made for {another} opens read-only."
        )
        self.game_data_combo.currentIndexChanged.connect(self._on_game_data_changed)
        row.addWidget(self.game_data_combo)
        row.addStretch(1)
        layout.addLayout(row)
        note = QLabel(
            "Leave this on Auto unless a file names the wrong game. Applies to the open "
            "scenario now and to every scenario opened later."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)
        # GH #32: whether the Units catalog lists GAIA-only objects for a player owner.
        layout.addWidget(QLabel("GAIA-only objects:"))
        self.gaia_only_hide_radio = QRadioButton("Hide from the catalog when the owner is a player")
        self.gaia_only_keep_radio = QRadioButton("Keep listed; placement still gives them to GAIA")
        tooltip = (
            "Wild animals, resources and trees can only belong to GAIA, as in the game's editor. "
            "Either way a placement gives them to GAIA, and no owner change gives one to a player."
        )
        gaia_only_group = QButtonGroup(tab)
        for radio in (self.gaia_only_hide_radio, self.gaia_only_keep_radio):
            radio.setToolTip(tooltip)
            gaia_only_group.addButton(radio)
            layout.addWidget(radio)
        if settings.get_gaia_only_catalog_mode() == "hide":
            self.gaia_only_hide_radio.setChecked(True)
        else:
            self.gaia_only_keep_radio.setChecked(True)
        # One radio's toggled covers both directions inside an exclusive group.
        self.gaia_only_hide_radio.toggled.connect(self._on_gaia_only_mode_toggled)
        layout.addStretch(1)
        return tab

    def _on_game_data_changed(self, index: int) -> None:
        self._window.set_game_data(self.game_data_combo.itemData(index))
        self._window._log_status(f"Game data: {self.game_data_combo.itemText(index)}")

    def _on_gaia_only_mode_toggled(self, hide: bool) -> None:
        settings.set_gaia_only_catalog_mode("hide" if hide else "keep")
        self._window.units_panel.refresh_owner_filter()
        shown = "hidden for a player owner" if hide else "listed for every owner"
        self._window._log_status(f"GAIA-only objects: {shown}")

    def _build_saving_tab(self) -> QWidget:
        """Autosave's four controls plus the .bak/.orig switch.

        Every widget is assigned to self, not left a bare local like the
        General tab's dark_checkbox: a bare local is unreachable from any
        GUI test, and these have to be drivable from one. Each value is set
        BEFORE its signal is connected, or merely constructing this dialog
        would write config.yaml -- the same trap the View menu's tick
        actions already document.

        No OK/Apply/Cancel to design around: this dialog is Close-only, so
        each handler persists and applies in one step, like
        _on_dark_mode_toggled. Combos rather than spin boxes for the two
        membership-gated values, which makes their off-list ValueError
        unreachable from the UI.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.autosave_enabled_check = QCheckBox("Autosave on a timer")
        self.autosave_enabled_check.setToolTip(
            "Writes a recovery snapshot to its own rotating slot, never to "
            "the file you have open, and never clears the unsaved-changes "
            "marker. A tick is skipped entirely while a document is clean, "
            "unchanged since its last autosave, or mid-stroke."
        )
        self.autosave_enabled_check.setChecked(settings.get_autosave_enabled())
        self.autosave_enabled_check.toggled.connect(self._on_autosave_enabled_toggled)
        layout.addWidget(self.autosave_enabled_check)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Autosave every:"))
        self.autosave_interval_combo = QComboBox()
        for minutes in settings.AUTOSAVE_INTERVAL_CHOICES:
            self.autosave_interval_combo.addItem(f"{minutes} min", minutes)
        self.autosave_interval_combo.setCurrentIndex(
            settings.AUTOSAVE_INTERVAL_CHOICES.index(settings.get_autosave_interval_min())
        )
        self.autosave_interval_combo.currentIndexChanged.connect(self._on_autosave_interval_changed)
        interval_row.addWidget(self.autosave_interval_combo)
        interval_row.addStretch(1)
        layout.addLayout(interval_row)

        retention_row = QHBoxLayout()
        retention_row.addWidget(QLabel("Keep per document:"))
        self.autosave_retention_combo = QComboBox()
        for count in settings.AUTOSAVE_RETENTION_CHOICES:
            self.autosave_retention_combo.addItem(f"{count} autosave{'s' if count != 1 else ''}", count)
        self.autosave_retention_combo.setCurrentIndex(
            settings.AUTOSAVE_RETENTION_CHOICES.index(settings.get_autosave_retention())
        )
        self.autosave_retention_combo.currentIndexChanged.connect(self._on_autosave_retention_changed)
        retention_row.addWidget(self.autosave_retention_combo)
        retention_row.addStretch(1)
        layout.addLayout(retention_row)

        location_row = QHBoxLayout()
        location_row.addWidget(QLabel("Autosaves go:"))
        self.autosave_location_combo = QComboBox()
        self.autosave_location_combo.addItem("In the autosave folder", "central")
        self.autosave_location_combo.addItem("Beside the scenario file", "sidecar")
        self.autosave_location_combo.setToolTip(
            "Beside the file falls back to the autosave folder below for any "
            "document it can't serve: an untitled one, a mods/subscribed/ path, "
            "one in the game's own profile folder, or a shipped template."
        )
        self.autosave_location_combo.setCurrentIndex(
            settings.AUTOSAVE_LOCATION_CHOICES.index(settings.get_autosave_location())
        )
        self.autosave_location_combo.currentIndexChanged.connect(self._on_autosave_location_changed)
        location_row.addWidget(self.autosave_location_combo)
        location_row.addStretch(1)
        layout.addLayout(location_row)

        # GH #127: where central slots go. Read-only text, so every change goes through the setter's refusal.
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Autosave folder:"))
        self.autosave_dir_edit = QLineEdit()
        self.autosave_dir_edit.setReadOnly(True)
        self.autosave_dir_edit.setPlaceholderText(f"{autosave.autosave_dir()} (default)")
        self.autosave_dir_edit.setText(settings.get_autosave_dir())
        self.autosave_dir_edit.setToolTip(settings.get_autosave_dir() or self.autosave_dir_edit.placeholderText())
        folder_row.addWidget(self.autosave_dir_edit, stretch=1)
        self.autosave_dir_browse_button = QPushButton("Browse…")
        self.autosave_dir_browse_button.clicked.connect(self._browse_autosave_dir)
        folder_row.addWidget(self.autosave_dir_browse_button)
        self.autosave_dir_default_button = QPushButton("Use Default")
        self.autosave_dir_default_button.clicked.connect(lambda: self._on_autosave_dir_picked(""))
        folder_row.addWidget(self.autosave_dir_default_button)
        layout.addLayout(folder_row)
        self.autosave_dir_status_label = QLabel()
        self.autosave_dir_status_label.setWordWrap(True)
        layout.addWidget(self.autosave_dir_status_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        self.backups_enabled_check = QCheckBox("Write .bak and .orig backups on save")
        self.backups_enabled_check.setToolTip(
            ".bak refreshes on every save that changes bytes; .orig is a "
            "one-time pristine snapshot of the file as it was first opened. "
            "Saves into the game's own profile folder always write both."
        )
        self.backups_enabled_check.setChecked(settings.get_backups_enabled())
        self.backups_enabled_check.toggled.connect(self._on_backups_enabled_toggled)
        layout.addWidget(self.backups_enabled_check)

        layout.addStretch(1)
        return tab

    def _on_autosave_enabled_toggled(self, enabled: bool) -> None:
        settings.set_autosave_enabled(enabled)
        self._window._reset_autosave_timer()
        self._window._log_status(f"Autosave: {'on' if enabled else 'off'}")

    def _on_autosave_interval_changed(self, index: int) -> None:
        minutes = self.autosave_interval_combo.itemData(index)
        settings.set_autosave_interval_min(minutes)
        self._window._reset_autosave_timer()
        self._window._log_status(f"Autosave interval: {minutes} min")

    def _on_autosave_retention_changed(self, index: int) -> None:
        # Retention and location both take effect on the next tick, so
        # neither needs _reset_autosave_timer().
        count = self.autosave_retention_combo.itemData(index)
        settings.set_autosave_retention(count)
        self._window._log_status(f"Autosaves kept per document: {count}")

    def _on_autosave_location_changed(self, index: int) -> None:
        location = self.autosave_location_combo.itemData(index)
        settings.set_autosave_location(location)
        self._window._log_status(f"Autosave location: {location}")

    def _browse_autosave_dir(self) -> None:
        start_dir = settings.get_autosave_dir() or str(autosave.autosave_dir())
        path = QFileDialog.getExistingDirectory(self, "Select autosave folder", start_dir)
        if path:
            self._on_autosave_dir_picked(path)

    def _set_autosave_dir_status(self, message: str, ok: bool) -> None:
        self._autosave_dir_status_ok = ok
        color = status_color(ok, on_tab_page=True)
        self.autosave_dir_status_label.setStyleSheet(f"color: {color};")
        self.autosave_dir_status_label.setText(message)

    def _on_autosave_dir_picked(self, path: str) -> None:
        """Browse or Use Default (""). Takes effect on the next tick; a refused
        folder leaves the setting as it was."""
        try:
            settings.set_autosave_dir(path)
        except ValueError as e:
            self._set_autosave_dir_status(str(e), ok=False)
            self._window._log_status(f"Autosave folder refused: {e}", level=debug_log.WARNING)
            return
        stored = settings.get_autosave_dir()
        self.autosave_dir_edit.setText(stored)
        shown = stored or f"{autosave.autosave_dir()} (default)"
        self.autosave_dir_edit.setToolTip(shown)
        self._set_autosave_dir_status(
            "Existing autosaves stay where they are and are still listed in File > Recover from Autosave.",
            ok=True,
        )
        self._window._log_status(f"Autosave folder: {shown}")

    def _on_backups_enabled_toggled(self, enabled: bool) -> None:
        settings.set_backups_enabled(enabled)
        self._window._log_status(f"Save backups: {'on' if enabled else 'off'}")

    def _build_keybinds_tab(self) -> QWidget:
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)

        rows_widget = QWidget()
        grid = QGridLayout(rows_widget)
        # The wrapped action labels take the slack, not the key boxes (TASK-498).
        grid.setColumnStretch(0, 1)
        # A key box shows every default sequence whole. Room for three modifiers would break 520 px.
        default_keys = [default for _aid, _label, default in settings.REBINDABLE_ACTIONS]

        self._keybind_edits: dict[str, QKeySequenceEdit] = {}
        # GH #116's filter: per section, its divider (None for the first), header and rows.
        self._keybind_sections: list[tuple[QFrame | None, QLabel, list[tuple[str, list[QWidget]]]]] = []
        row = self._build_wheel_rows(grid)
        # Seeded with the wheel section, so the first key section below draws a divider above itself.
        current_section = "wheel"
        for action_id, label, _default in settings.REBINDABLE_ACTIONS:
            section = action_id.split("_", 1)[0]
            if section != current_section:
                divider = None
                if current_section is not None:
                    divider = QFrame()
                    divider.setFrameShape(QFrame.HLine)
                    divider.setFrameShadow(QFrame.Sunken)
                    grid.addWidget(divider, row, 0, 1, 4)
                    row += 1
                section_label = QLabel(f"<b>{settings.KEYBIND_SECTION_TITLES.get(section, section.title())}</b>")
                grid.addWidget(section_label, row, 0, 1, 4)
                row += 1
                current_section = section
                self._keybind_sections.append((divider, section_label, []))

            # A mode-scoped row names its modes (GH #267), and the search below matches them.
            action_label = QLabel(label + settings.keybind_mode_tag(action_id))
            action_label.setWordWrap(True)
            grid.addWidget(action_label, row, 0)

            edit = _KeyBox(QKeySequence(settings.get_keybind(action_id)), default_keys)
            edit.keySequenceChanged.connect(
                lambda seq, aid=action_id: self._on_keybind_changed(aid, seq)
            )
            # QKeySequenceEdit has no setAlignment of its own -- it wraps an
            # internal QLineEdit, which does.
            line_edit = edit.findChild(QLineEdit)
            if line_edit is not None:
                line_edit.setAlignment(Qt.AlignCenter)
            grid.addWidget(edit, row, 1)
            self._keybind_edits[action_id] = edit

            default_btn = _TextWidthButton("Default")
            default_btn.clicked.connect(lambda _checked, aid=action_id: self._reset_keybind(aid))
            grid.addWidget(default_btn, row, 2)

            clear_btn = _TextWidthButton("Clear")
            clear_btn.clicked.connect(lambda _checked, aid=action_id: self._keybind_edits[aid].clear())
            grid.addWidget(clear_btn, row, 3)

            self._keybind_sections[-1][2].append((action_id, [action_label, edit, default_btn, clear_btn]))
            row += 1

        grid.setRowStretch(row, 1)

        self._keybind_filter_edit = QLineEdit()
        self._keybind_filter_edit.setPlaceholderText("Search actions or keys")
        self._keybind_filter_edit.setClearButtonEnabled(True)
        self._keybind_filter_edit.textChanged.connect(self._filter_keybinds)
        outer_layout.addWidget(self._keybind_filter_edit)
        # Unbolded: the section-header test collects <b> labels.
        self._keybind_no_match_label = QLabel("No keybinds match.")
        self._keybind_no_match_label.setHidden(True)
        outer_layout.addWidget(self._keybind_no_match_label)

        # Vertical only, so the rows' minimum width widens the dialog (TASK-375).
        scroll = _VerticalScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_widget)
        outer_layout.addWidget(scroll)

        # Feedback surface for a keybind-collision auto-clear (see
        # _on_keybind_changed) -- deliberately unbolded, since
        # test_keybinds_tab_renders_menu_section_headers collects section
        # headers by filtering QLabel.text().startswith("<b>").
        self._keybind_warning_label = QLabel("")
        self._keybind_warning_label.setWordWrap(True)
        outer_layout.addWidget(self._keybind_warning_label)

        return tab

    def _build_wheel_rows(self, grid: QGridLayout) -> int:
        """The Mouse Wheel section that opens the Keybinds tab: one dropdown of
        settings.WHEEL_ACTIONS per settings.WHEEL_SLOTS slot, with no Clear
        button since "Nothing" is the unbound state. Returns the next grid row.

        Its rows join _keybind_sections for the search, with ids "wheel_<slot>"."""
        self._wheel_combos: dict[str, QComboBox] = {}
        self._wheel_header = QLabel("<b>Mouse Wheel</b>")
        grid.addWidget(self._wheel_header, 0, 0, 1, 4)
        rows: list[tuple[str, list[QWidget]]] = []
        self._keybind_sections.append((None, self._wheel_header, rows))
        bindings = settings.get_wheel_bindings()
        row = 1
        for slot in settings.WHEEL_SLOTS:
            slot_label = QLabel(settings.WHEEL_SLOT_LABELS[slot])
            grid.addWidget(slot_label, row, 0, 1, 4)
            row += 1

            # NoWheel: scrolling the tab over a row must not rebind it.
            combo = _WholeLabelComboBox()
            # Whole at any width: the slot label's own line above leaves it the row (TASK-498).
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            for i, action in enumerate(settings.WHEEL_ACTIONS):
                combo.addItem(settings.WHEEL_ACTION_LABELS[action], action)
                combo.setItemData(i, settings.WHEEL_ACTION_HINTS[action], Qt.ToolTipRole)
            combo.setCurrentIndex(settings.WHEEL_ACTIONS.index(bindings[slot]))
            combo.currentIndexChanged.connect(lambda _index, s=slot: self._on_wheel_binding_changed(s))
            # Spans the label, key-box and Default columns, and Default takes Clear's.
            grid.addWidget(combo, row, 0, 1, 3)
            self._wheel_combos[slot] = combo

            default_btn = _TextWidthButton("Default")
            default_btn.clicked.connect(lambda _checked, s=slot: self._reset_wheel_binding(s))
            grid.addWidget(default_btn, row, 3)

            rows.append((f"wheel_{slot}", [slot_label, combo, default_btn]))
            row += 1

        # Unbolded (the section-header test collects <b> labels) and in no row's
        # widget list, so _filter_keybinds hides it with the header.
        self._wheel_note = QLabel(
            "Two rows may share an action. Any other modifier combination zooms. "
            "Holding Shift or Alt for a tool's own modifier (a straight wall run, off-grid placement) "
            "uses that row. Some desktops take Alt + Wheel for themselves. "
            "Brush size and the Rotate choices do nothing where they have nothing to act on."
        )
        self._wheel_note.setWordWrap(True)
        grid.addWidget(self._wheel_note, row, 0, 1, 4)
        return row + 1

    def _on_wheel_binding_changed(self, slot: str) -> None:
        action = self._wheel_combos[slot].currentData()
        settings.set_wheel_binding(slot, action)
        self._window.map_view.set_wheel_bindings(settings.get_wheel_bindings())
        self._window._log_status(
            f"Wheel binding changed: {settings.WHEEL_SLOT_LABELS[slot]} -> {settings.WHEEL_ACTION_LABELS[action]}"
        )
        # A search by action would otherwise keep showing rows by their old binding.
        self._filter_keybinds(self._keybind_filter_edit.text())

    def _reset_wheel_binding(self, slot: str) -> None:
        # Drives the combo, so _on_wheel_binding_changed() does the persist + apply + log.
        default = settings.DEFAULT_WHEEL_BINDINGS[slot]
        self._wheel_combos[slot].setCurrentIndex(settings.WHEEL_ACTIONS.index(default))

    def _keybind_row_binding(self, row_id: str) -> str:
        """A Keybinds row's second search string: its key sequence, or for a
        wheel row ("wheel_<slot>") the action its dropdown shows."""
        edit = self._keybind_edits.get(row_id)
        if edit is not None:
            return edit.keySequence().toString()
        return self._wheel_combos[row_id.removeprefix("wheel_")].currentText()

    def _filter_keybinds(self, text: str) -> None:
        """Hides the Keybinds rows whose action label and bound key both miss
        `text` (case-insensitive), and a section's header and divider once all
        of its rows are hidden. The first shown section never shows a divider."""
        needle = text.strip().casefold()
        any_shown = False
        for divider, header, rows in self._keybind_sections:
            section_shown = False
            for action_id, widgets in rows:
                key = self._keybind_row_binding(action_id)
                shown = not needle or needle in widgets[0].text().casefold() or needle in key.casefold()
                for widget in widgets:
                    widget.setHidden(not shown)
                section_shown = section_shown or shown
            header.setHidden(not section_shown)
            if divider is not None:
                divider.setHidden(not (section_shown and any_shown))
            any_shown = any_shown or section_shown
        self._wheel_note.setHidden(self._wheel_header.isHidden())
        self._keybind_no_match_label.setHidden(any_shown)

    @staticmethod
    def _keybind_names(action_ids: list[str]) -> str:
        """The actions' tagged labels as "A", "A and B" or "A, B and C", for the warnings."""
        labels = [settings.keybind_display_label(action_id) for action_id in action_ids]
        return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"

    def _on_keybind_changed(self, action_id: str, key_sequence: QKeySequence) -> None:
        text = key_sequence.toString()
        cleared = settings.set_keybind(action_id, text)
        self._window.apply_keybind(action_id)
        self._window._log_status(f"Keybind changed: {action_id} -> {text or '(cleared)'}")
        if not cleared:
            self._keybind_warning_label.clear()
        else:
            for cleared_action_id in cleared:
                self._window.apply_keybind(cleared_action_id)
                cleared_edit = self._keybind_edits.get(cleared_action_id)
                if cleared_edit is not None:
                    cleared_edit.blockSignals(True)
                    cleared_edit.clear()
                    cleared_edit.blockSignals(False)
            which = "that binding has" if len(cleared) == 1 else "those bindings have"
            self._keybind_warning_label.setText(
                f"'{text}' was already assigned to {self._keybind_names(cleared)} -- {which} been cleared."
            )
        # A search by key would otherwise keep showing rows by their old binding.
        self._filter_keybinds(self._keybind_filter_edit.text())

    def _reset_keybind(self, action_id: str) -> None:
        default = settings.get_default_keybind(action_id)
        holders = settings.keybind_conflicts(default, action_id)
        if holders:
            # Any other action holding this default where this one is live is
            # necessarily off its own default (test_default_keybinds_are_pairwise_
            # conflict_free pins that the shipped defaults are pairwise
            # conflict-free), so a deliberate binding is at stake. Restoring a
            # default must not outrank it. Typing the sequence in is still an
            # explicit override.
            holder_label = self._keybind_names(holders)
            self._keybind_warning_label.setText(
                f"'{default}' is assigned to {holder_label} -- "
                f"{settings.get_action_label(action_id)}'s default was not restored."
            )
            self._window._log_status(
                f"Keybind unchanged: {action_id} ('{default}' held by {holder_label})"
            )
            return
        # setKeySequence() emits keySequenceChanged, so _on_keybind_changed()
        # handles the actual persist + apply + log.
        self._keybind_edits[action_id].setKeySequence(QKeySequence(default))

    def _add_install_section(self, layout: QVBoxLayout) -> None:
        """Appends the AoE2DE install-path controls into the given layout --
        folded into the General tab (moved out of its own "Game Resources"
        tab) since this is a one-time setup step, not a recurring setting
        someone flips back and forth."""
        install_label = QLabel("AoE2DE install path (optional, for real terrain colors):")
        install_label.setWordWrap(True)
        layout.addWidget(install_label)

        install_row = QHBoxLayout()
        self.install_path_edit = QLineEdit()
        existing_install = asset_source.get_install_path()
        if existing_install is not None:
            self.install_path_edit.setText(str(existing_install))
        install_row.addWidget(self.install_path_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_install_path)
        install_row.addWidget(browse_btn)
        layout.addLayout(install_row)

        install_buttons = QHBoxLayout()
        load_install_btn = QPushButton("Load")
        load_install_btn.clicked.connect(self.load_install_path)
        install_buttons.addWidget(load_install_btn, stretch=1)
        # TASK-101: Load refuses an empty field, so this is the one way back to no install.
        self.install_clear_button = QPushButton("Clear")
        self.install_clear_button.setToolTip("Forget the install path set here; terrain goes back to flat colors.")
        self.install_clear_button.clicked.connect(self.clear_install_path)
        self.install_clear_button.setEnabled(existing_install is not None)
        install_buttons.addWidget(self.install_clear_button)
        layout.addLayout(install_buttons)

        self.install_status_label = QLabel("")
        self.install_status_label.setWordWrap(True)
        layout.addWidget(self.install_status_label)

        # Reflects whatever's actually configured right now (dialog-open
        # time), not just the outcome of the last "Load" click in this
        # session -- e.g. a path set via AOE2DE_INSTALL_PATH or a prior
        # session's config.yaml would otherwise leave this blank forever.
        if existing_install is not None:
            ok, message = asset_source.validate_install_path(existing_install)
            self._set_install_status(message, ok=ok)
        else:
            self._set_install_status("No install path configured -- using flat fallback colors.", ok=False)

    def browse_install_path(self) -> None:
        start_dir = self.install_path_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select AoE2DE install folder", start_dir)
        if path:
            self.install_path_edit.setText(path)

    def load_install_path(self) -> None:
        text = self.install_path_edit.text().strip()
        if not text:
            self._set_install_status("Enter or browse to an AoE2DE install path first.", ok=False)
            return

        ok, message = asset_source.validate_install_path(Path(text))
        if not ok:
            self._set_install_status(message, ok=False)
            self._window._log_status(f"Install path rejected ({text}): {message}", level=debug_log.WARNING)
            return

        path = Path(text)
        asset_source.set_install_path_override(path)
        asset_source.save_install_path_to_config(path)
        self._set_install_status(message, ok=True)
        self._window._log_status(f"Install path set to {path}: {message}{_sync_hi_res()}")
        self._after_install_change()

    def clear_install_path(self) -> None:
        """Drops the override and the saved path, so the art caches clear with it."""
        asset_source.save_install_path_to_config(None)
        asset_source.set_install_path_override(None)
        remaining = asset_source.get_install_path()  # only AOE2DE_INSTALL_PATH can still set one
        if remaining is None:
            self.install_path_edit.clear()
            self._set_install_status("No install path configured -- using flat fallback colors.", ok=False)
            self._window._log_status(f"Install path cleared{_sync_hi_res()}")
        else:
            self.install_path_edit.setText(str(remaining))
            self._set_install_status(f"Saved path cleared; AOE2DE_INSTALL_PATH still sets {remaining}.", ok=True)
            self._window._log_status(f"Install path cleared; AOE2DE_INSTALL_PATH still sets {remaining}{_sync_hi_res()}")
        self._after_install_change()

    def _after_install_change(self) -> None:
        self.install_clear_button.setEnabled(asset_source.get_install_path() is not None)
        self._update_x2_art_label()
        terrain_catalog.install_changed()  # GH #271: a data mod's terrains and names follow the install's strings
        self._window.refresh_map()
        if game_dataset.data_mod() is not None:
            self._window.terrain_panel.reload_catalog()  # fresh rows, so fresh swatches too
        else:
            # The long-lived picker keeps its old swatches otherwise (see refresh_swatches()).
            self._window.terrain_panel.view.refresh_swatches()
        # The pending pick is sticky, so its card would otherwise wait for the next pick (TASK-337).
        self._window.units_panel.refresh_thumbnails()
        self._window.invalidate_ai_choices()  # GH #126: Personality lists this install's AIs
        self._refresh_profile_section()  # auto-detect derives the profile from the install
        self._window.data_mod_menu.install_changed()  # GH #58: its tables were read against the old install

    def _set_install_status(self, message: str, ok: bool) -> None:
        self._install_status_ok = ok
        color = status_color(ok, on_tab_page=True)
        self.install_status_label.setStyleSheet(f"color: {color};")
        self.install_status_label.setText(message)

    # Rows the scanned-folders list shows before it scrolls.
    _PROFILE_LIST_ROWS = 5

    def _add_profile_section(self, layout: QVBoxLayout) -> None:
        """The AoE2:DE profile folder row (GH #126), same shape as the install
        row, plus a read-only list of every folder content_roots scans."""
        profile_label = QLabel("AoE2DE profile folder (optional, for custom AIs from your profile and mods):")
        profile_label.setWordWrap(True)
        layout.addWidget(profile_label)

        profile_row = QHBoxLayout()
        self.profile_path_edit = QLineEdit()
        profile_row.addWidget(self.profile_path_edit, stretch=1)
        self.profile_browse_button = QPushButton("Browse…")
        self.profile_browse_button.clicked.connect(self.browse_profile_path)
        profile_row.addWidget(self.profile_browse_button)
        layout.addLayout(profile_row)

        button_row = QHBoxLayout()
        self.profile_load_button = QPushButton("Load")
        self.profile_load_button.clicked.connect(self.load_profile_path)
        button_row.addWidget(self.profile_load_button)
        self.profile_auto_button = QPushButton("Use auto-detect")
        self.profile_auto_button.setToolTip("Forget the folder set here and find it from the install again.")
        self.profile_auto_button.clicked.connect(self.use_auto_profile_path)
        button_row.addWidget(self.profile_auto_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.profile_status_label = QLabel("")
        self.profile_status_label.setWordWrap(True)
        layout.addWidget(self.profile_status_label)

        layout.addWidget(QLabel("Folders scanned for content:"))
        self.profile_roots_list = QListWidget()
        self.profile_roots_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.profile_roots_list.setFocusPolicy(Qt.NoFocus)
        # Long paths scroll sideways inside the list instead of widening the dialog.
        self.profile_roots_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.profile_roots_list.setWordWrap(False)
        layout.addWidget(self.profile_roots_list)
        self.profile_skipped_label = QLabel("")
        self.profile_skipped_label.setWordWrap(True)
        layout.addWidget(self.profile_skipped_label)

        self._refresh_profile_section(refill_field=True)

    def _add_mod_art_section(self, layout: QVBoxLayout) -> None:
        """Every scanned mod that ships art, in the game's mod order (top
        wins), ticked when View > Layers > Mod Art draws it (TASK-378). A mod
        the game's mod list disables is listed too, marked and unticked until
        ticked here (GH #264). A tick applies at once; an unticked mod stays
        off even if it is reinstalled. Rescan mods re-reads every mod folder (TASK-438)."""
        label_row = QHBoxLayout()
        label = QLabel("Mod art drawn by View > Layers > Mod Art (the top mod wins where two ship one file):")
        label.setWordWrap(True)
        label_row.addWidget(label, stretch=1)
        self.mod_rescan_button = QPushButton("Rescan mods")
        self.mod_rescan_button.setToolTip(
            "Re-reads every mod folder, for a mod the game updated while DEscape was open.\n"
            "A mod enabled, disabled or reordered in the game is picked up without this."
        )
        self.mod_rescan_button.clicked.connect(self.rescan_mods)
        label_row.addWidget(self.mod_rescan_button, alignment=Qt.AlignTop)
        layout.addLayout(label_row)
        self.mod_art_list = QListWidget()
        self.mod_art_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.mod_art_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mod_art_list.setWordWrap(False)
        self.mod_art_list.itemChanged.connect(self._on_mod_art_item_changed)
        layout.addWidget(self.mod_art_list)
        self._refresh_mod_art_list()

    def _refresh_mod_art_list(self) -> None:
        view = self.mod_art_list
        view.blockSignals(True)
        view.clear()
        for mod in mod_graphics.mods_with_art():
            item = QListWidgetItem(mod.title if mod.in_game else f"{mod.title} (disabled in game)")
            # UserRole stays the bare status path; +1 whether the game enables it, +2 the plain title.
            item.setData(Qt.UserRole, mod.status_path)
            item.setData(Qt.UserRole + 1, mod.in_game)
            item.setData(Qt.UserRole + 2, mod.title)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if mod.drawn else Qt.Unchecked)
            order = "not in the game's mod list" if mod.priority is None else f"priority {mod.priority} in the game's mod list"
            disabled = "" if mod.in_game else "Disabled in the game's mod list, drawn here only when ticked\n"
            item.setToolTip(
                f"{disabled}{mod.status_path}, {order}\n"
                f"{mod.sld} .sld and {mod.smx} .smx sprites, {mod.textures} terrain textures"
            )
            view.addItem(item)
        if view.count() == 0:
            placeholder = QListWidgetItem("No mod in the profile folder above ships art.")
            placeholder.setFlags(Qt.NoItemFlags)
            view.addItem(placeholder)
        view.blockSignals(False)
        self._fit_list_height(view)

    def _on_mod_art_item_changed(self, item: QListWidgetItem) -> None:
        status_path = item.data(Qt.UserRole)
        if status_path is None:
            return
        drawn = item.checkState() == Qt.Checked
        if item.data(Qt.UserRole + 1):  # enabled in the game: the tick edits its opt-out
            excluded = set(settings.get_mod_art_excluded())
            if drawn:
                excluded.discard(status_path)
            else:
                excluded.add(status_path)
            settings.set_mod_art_excluded(excluded)
        else:  # disabled in the game: its opt-in (GH #264)
            included = set(settings.get_mod_art_included())
            if drawn:
                included.add(status_path)
            else:
                included.discard(status_path)
            settings.set_mod_art_included(included)
        title = item.data(Qt.UserRole + 2)
        if self._window._apply_mod_art_change(f"Mod art from {title}: {'on' if drawn else 'off'}"):
            # The game's mod list changed meanwhile (TASK-438). Deferred: this slot's item would be deleted under it.
            QTimer.singleShot(0, self._refresh_mod_art_list)

    def rescan_mods(self) -> None:
        """Rescan mods: forgets the scanned mod folders, as a profile folder change does."""
        self._window._apply_mod_art_change("Rescanned mods", rescan=True)
        self._refresh_profile_section()
        self._window.invalidate_ai_choices()

    def _add_data_mod_section(self, layout: QVBoxLayout) -> None:
        """GH #265: two Units catalog options for a data mod in effect, and GH #271's Terrain panel one, applied at once."""
        layout.addWidget(QLabel("Data mods (File > Data Mod):"))
        self.data_mod_hide_unchanged_check = QCheckBox("Hide unchanged objects in the Units catalog")
        self.data_mod_hide_unchanged_check.setToolTip(
            "Lists only the objects the data mod adds or changes: new, repurposed, renamed, "
            "or with other art or stats than the game's own. An object picked with the "
            "Eyedropper shows while it is the pick. Trigger pickers, Find and Map Analysis "
            "still list everything."
        )
        self.data_mod_hide_unchanged_check.setChecked(settings.get_data_mod_hide_unchanged())
        self.data_mod_hide_unchanged_check.toggled.connect(self._on_data_mod_hide_unchanged_toggled)
        layout.addWidget(self.data_mod_hide_unchanged_check)
        self.data_mod_label_check = QCheckBox('Show "(data mod)" after mod-only objects')
        self.data_mod_label_check.setToolTip(
            "In the Units catalog, the placing card, the inspector and the pickers that name objects."
        )
        self.data_mod_label_check.setChecked(settings.get_data_mod_label_shown())
        self.data_mod_label_check.toggled.connect(self._on_data_mod_label_toggled)
        layout.addWidget(self.data_mod_label_check)
        self.data_mod_terrain_group_check = QCheckBox("List the data mod's own terrains in one group")
        self.data_mod_terrain_group_check.setToolTip(
            'In the Terrain panel, a "Mod terrains" group at the top holds every terrain the data mod '
            "adds or changes, instead of each in its category."
        )
        self.data_mod_terrain_group_check.setChecked(settings.get_data_mod_terrain_group())
        self.data_mod_terrain_group_check.toggled.connect(self._on_data_mod_terrain_group_toggled)
        layout.addWidget(self.data_mod_terrain_group_check)

    def _on_data_mod_hide_unchanged_toggled(self, hide: bool) -> None:
        settings.set_data_mod_hide_unchanged(hide)
        self._window.units_panel.refresh_owner_filter()
        shown = "hidden from" if hide else "listed in"
        self._window._log_status(f"Objects a data mod did not change: {shown} the Units catalog")

    def _on_data_mod_terrain_group_toggled(self, grouped: bool) -> None:
        settings.set_data_mod_terrain_group(grouped)
        self._window.terrain_panel.reload_catalog()
        if self._window.mode == "terrain":
            self._window._apply_left_pane_floor()
        where = "in one group" if grouped else "in their categories"
        self._window._log_status(f"A data mod's own terrains: listed {where} in the Terrain panel")

    def _on_data_mod_label_toggled(self, shown: bool) -> None:
        settings.set_data_mod_label_shown(shown)
        object_catalog.objects.cache_clear()
        self._window.units_panel.rename_catalog()
        self._window._update_unit_inspector_from_selection()
        self._window._log_status(f'"(data mod)" label: {"on" if shown else "off"}')

    def _fit_profile_list_height(self) -> None:
        self._fit_list_height(self.profile_roots_list)

    def _fit_list_height(self, view: QListWidget) -> None:
        row = view.sizeHintForRow(0) if view.count() else view.fontMetrics().height()
        bar = view.horizontalScrollBar().sizeHint().height()
        view.setFixedHeight(row * self._PROFILE_LIST_ROWS + bar + 2 * view.frameWidth())

    def _refresh_profile_section(self, *, refill_field: bool = False) -> None:
        """Status line, field state and scanned list from the live lookups.
        The field is only overwritten on open, or when nothing explicit is set."""
        source, raw = content_roots.profile_source()
        profile = content_roots.profile_path()
        from_env = source in (content_roots.SOURCE_ENV, content_roots.SOURCE_ENV_INVALID)
        if refill_field or from_env or source in (content_roots.SOURCE_AUTO, content_roots.SOURCE_NONE):
            shown = raw if raw is not None else (str(profile) if profile is not None else "")
            self.profile_path_edit.setText(shown)
        for widget in (self.profile_path_edit, self.profile_browse_button, self.profile_load_button):
            widget.setEnabled(not from_env)
        # Only a config.yaml setting has anything to forget.
        self.profile_auto_button.setEnabled(source in (content_roots.SOURCE_CONFIG, content_roots.SOURCE_CONFIG_INVALID))

        env = content_roots.PROFILE_ENV_VAR
        if source == content_roots.SOURCE_ENV:
            self._set_profile_status(f"Set by {env}, which overrides this field.", ok=True)
        elif source == content_roots.SOURCE_ENV_INVALID:
            self._set_profile_status(f"{env} points at {raw}, which is not a folder, so no profile is scanned.", ok=False)
        elif source == content_roots.SOURCE_CONFIG:
            self._set_profile_status("From config.yaml.", ok=True)
        elif source == content_roots.SOURCE_CONFIG_INVALID:
            self._set_profile_status(f"config.yaml points at {raw}, which is not a folder, so no profile is scanned.", ok=False)
        elif source == content_roots.SOURCE_AUTO:
            where = "your user folder" if sys.platform == "win32" else "the install"
            self._set_profile_status(f"Auto-detected from {where}.", ok=True)
        else:
            self._set_profile_status("No profile folder found: only the install's own content is offered.", ok=False)

        self.profile_roots_list.clear()
        for root in content_roots.content_roots():
            item = QListWidgetItem(f"{root.label}: {root.path}")
            item.setToolTip(str(root.path))
            self.profile_roots_list.addItem(item)
        if self.profile_roots_list.count() == 0:
            self.profile_roots_list.addItem("Nothing yet: set the install path on the General tab.")
        skipped = content_roots.skipped_mods(profile) if profile is not None else []
        # Every profile id's disabled mods, art or not; the Mod art list shows the active id's art mods.
        if len(skipped) == 1:
            note = (
                "1 mod disabled in the game's mod list is skipped; if it is in the profile the game last used "
                "and ships art, it is listed below, off until ticked."
            )
        elif skipped:
            note = (
                f"{len(skipped)} mods disabled in the game's mod list are skipped; any in the profile the game "
                "last used that ship art are listed below, off until ticked."
            )
        else:
            note = ""
        self.profile_skipped_label.setText(note)
        self.profile_skipped_label.setVisible(bool(skipped))
        self._fit_profile_list_height()
        # Built after this section on open; refreshed with it on every later profile or install change.
        if hasattr(self, "mod_art_list"):
            self._refresh_mod_art_list()

    def browse_profile_path(self) -> None:
        start_dir = self.profile_path_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select AoE2DE profile folder", start_dir)
        if path:
            self.profile_path_edit.setText(path)

    def load_profile_path(self) -> None:
        text = self.profile_path_edit.text().strip()
        if not text:
            self._set_profile_status("Enter or browse to a profile folder first.", ok=False)
            return
        path = Path(text)
        if not path.is_dir() or not content_roots.has_profile_ids(path):
            message = (
                f"{text} is not a profile folder: it holds no numeric per-account folder "
                "(the one named 0 or after your Steam ID). Nothing saved."
            )
            self._set_profile_status(message, ok=False)
            self._window._log_status(f"Profile folder rejected ({text})", level=debug_log.WARNING)
            return
        asset_source.save_profile_path_to_config(path)
        self._window._apply_mod_art_change(f"Profile folder set to {path}", rescan=True)
        self._refresh_profile_section(refill_field=True)
        self._window.invalidate_ai_choices()  # Personality rescans the new folder's AIs

    def use_auto_profile_path(self) -> None:
        asset_source.save_profile_path_to_config(None)
        self._window._apply_mod_art_change("Profile folder: auto-detect", rescan=True)
        self._refresh_profile_section(refill_field=True)
        self._window.invalidate_ai_choices()

    def _set_profile_status(self, message: str, ok: bool) -> None:
        self._profile_status_ok = ok
        self.profile_status_label.setStyleSheet(f"color: {status_color(ok, on_tab_page=True)};")
        self.profile_status_label.setText(message)
