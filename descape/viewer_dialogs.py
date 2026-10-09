"""App-chrome theming and the Help > Debug Log window.

_LIGHT_PALETTE moves with apply_theme() rather than being re-exported:
it is `global`-mutated there, so a re-export would not track it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QPalette,
)
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from descape import (
    debug_log,
    themes,
)

# DebugLogDialog's parent is annotated "ViewerWindow". `from __future__ import
# annotations` means that string is never evaluated, so this stays a
# type-checker-only import -- viewer.py imports this module, and a runtime
# import back would be a cycle.
if TYPE_CHECKING:
    from descape.viewer import ViewerWindow


# Cached the first time apply_theme() runs, before any themed palette is ever
# applied -- the `light` preset itself and the base every other preset is
# built on. Captured from the Fusion style itself (style().standardPalette()),
# not whatever native platform style/palette was active before this app ever
# touched it: apply_theme() always forces Fusion, so the "light" state should
# be Fusion's own default, not a different style's palette that would look
# inconsistent switching back and forth.
_LIGHT_PALETTE: QPalette | None = None

# The same trick for apply_ui_font(): the app's font before this app ever
# set one, captured on the first call, so clearing the setting restores the
# real platform default rather than a guess at it.
_DEFAULT_FONT: QFont | None = None


_ROLE_ENUMS: dict[str, QPalette.ColorRole] = {role_id: getattr(QPalette, qt) for role_id, qt in themes.ROLE_QT_NAMES.items()}
_BEVEL_ROLES = (QPalette.Light, QPalette.Midlight, QPalette.Mid, QPalette.Dark, QPalette.Shadow)
_GROUPS = (QPalette.Active, QPalette.Inactive, QPalette.Disabled)


def _build_dark_palette(base: QPalette) -> QPalette:
    """The pre-theme dark-mode recipe, the well-known Fusion-dark combination,
    built on a copy of `base` (Fusion's standard palette) rather than the old
    QPalette(), which copies whatever app palette is live. Identical for every
    role and group whenever apply_theme()'s setStyle() has just reset it."""
    colors = themes.PRESETS["dark"][1]
    palette = QPalette(base)
    for role_id, role in _ROLE_ENUMS.items():
        palette.setColor(role, QColor(colors[role_id]))
    palette.setColor(QPalette.BrightText, QColor(colors["bright_text"]))
    for role_id in themes.DERIVED_DISABLED_ROLES:
        palette.setColor(QPalette.Disabled, _ROLE_ENUMS[role_id], QColor(colors[f"disabled_{role_id}"]))
    return palette


def build_theme_palette(base: QPalette, preset_id: str, overrides: dict[str, str]) -> QPalette:
    """`base` is Fusion's standard palette. An unknown preset id is treated as
    themes.THEME_DEFAULT. See the plan-level rules in this module's
    apply_theme() docstring for what is derived."""
    if preset_id not in themes.PRESETS:
        preset_id = themes.THEME_DEFAULT
    legacy = preset_id in themes.LEGACY_PRESETS
    overrides = {role_id: hex_str for role_id, hex_str in overrides.items() if role_id in _ROLE_ENUMS}
    palette = _build_dark_palette(base) if preset_id == "dark" else QPalette(base)
    if legacy and not overrides:
        return palette

    preset = themes.PRESETS[preset_id][1]
    # Every group first, as the legacy recipe's group-less setColor does.
    for role_id, role in _ROLE_ENUMS.items():
        hex_str = overrides.get(role_id) or (None if legacy else preset[role_id])
        if hex_str is not None:
            palette.setColor(role, QColor(hex_str))

    def active(role_id: str) -> str:
        return palette.color(QPalette.Active, _ROLE_ENUMS[role_id]).name()

    derived = set(themes.DERIVED_DISABLED_ROLES) if not legacy else set(overrides)
    for role_id, background, amount in themes.DISABLED_BLENDS:
        if role_id in derived:
            dimmed = themes.blend(active(role_id), active(background), amount)
            palette.setColor(QPalette.Disabled, _ROLE_ENUMS[role_id], QColor(dimmed))
    if "highlighted_text" in derived:
        palette.setColor(
            QPalette.Disabled, QPalette.HighlightedText, palette.color(QPalette.Disabled, QPalette.Text)
        )

    if not legacy or "button" in overrides:
        # What the QPalette(QColor button) constructor derives, so a coloured
        # button never keeps Fusion's grey frame shades.
        bevels = QPalette(QColor(active("button")))
        for group in _GROUPS:
            for role in _BEVEL_ROLES:
                palette.setColor(group, role, bevels.color(group, role))
    return palette


def apply_theme(app: QApplication, preset_id: str, overrides: dict[str, str] | None = None) -> None:
    """App chrome only -- menus, dialogs, toolbars, the status bar. MapView's
    own colors (real per-tile terrain textures, unit dots, hover highlights)
    are untouched: they represent game data or are already dark, not UI
    styling that should shift with the theme. Called once at startup
    (main(), from settings.get_theme()/get_theme_colors()) and live from the
    Appearance settings tab -- safe to call repeatedly, setStyle("Fusion")
    is idempotent and _LIGHT_PALETTE is only ever captured once.

    `light` with no overrides is _LIGHT_PALETTE itself and `dark` with none is
    the legacy recipe. Anything else starts from Fusion's palette and sets
    each role in every group. Disabled text, button text, window text and
    highlight are then blended toward their background (highlighted text
    takes Disabled text), for every non-legacy preset and for any role
    overridden on a legacy one, whose other Disabled colours stay as they
    were. Frame shades follow the button colour for every non-legacy preset
    and for a button override."""
    global _LIGHT_PALETTE
    app.setStyle("Fusion")
    if _LIGHT_PALETTE is None:
        _LIGHT_PALETTE = app.style().standardPalette()
    if preset_id == "light" and not overrides:
        app.setPalette(_LIGHT_PALETTE)
        return
    app.setPalette(build_theme_palette(_LIGHT_PALETTE, preset_id, overrides or {}))


def apply_ui_font(app: QApplication, family: str, size: int | None) -> None:
    """App chrome only, apply_theme()'s sibling and same carve-out: the map
    view's own overlay text (ruler readout, distance-tick numbers, stacked-
    unit badges) is built from viewer_canvas.map_overlay_font() and does not
    follow this. Called once at startup (main()) and live from the
    Appearance settings tab -- safe to call repeatedly.

    `family` "" and `size` None both mean "platform default", and restore
    the baseline captured on the first call, the same cache-once trick
    apply_theme uses for _LIGHT_PALETTE. An unknown family is ignored
    rather than handed to Qt's silent substitution."""
    global _DEFAULT_FONT
    if _DEFAULT_FONT is None:
        _DEFAULT_FONT = QFont(app.font())
    font = QFont(_DEFAULT_FONT)
    if family and family in QFontDatabase().families():
        font.setFamily(family)
    if size is not None:
        font.setPointSize(size)
    app.setFont(font)


class DebugLogDialog(QDialog):
    """Read-only viewer over debug_log's in-memory buffer. A snapshot at open
    time (and after Refresh), not a live tail -- simplest thing that's useful
    for "what did the app just do," not meant as a full log console."""

    def __init__(self, parent: ViewerWindow):
        super().__init__(parent)
        self.setWindowTitle("Debug Log")
        self.resize(640, 400)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        # Family only, deliberately: a log wants columns, so monospace is
        # pinned against apply_ui_font(), but the size is left unset so it
        # still follows the user's chrome font size.
        self.text.setFont(QFont("Monospace"))
        self._refresh()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)

        btn_row = QHBoxLayout()
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.text, stretch=1)
        layout.addLayout(btn_row)
        layout.addWidget(buttons)

    def _refresh(self) -> None:
        self.text.setPlainText(debug_log.get_log_text())
        cursor = self.text.textCursor()
        cursor.movePosition(cursor.End)
        self.text.setTextCursor(cursor)

    def _clear(self) -> None:
        debug_log.clear()
        self._refresh()


class CrashReportDialog(QDialog):
    """Surfaces a crash report: read-only monospace dump text (same shape as
    DebugLogDialog) plus actions to inspect/save it. Not a QMessageBox --
    tests/conftest.py records that a message box blocks forever offscreen,
    which would hang the default-tier test suite; a QDialog is constructible
    and inspectable in a test without ever calling exec_()."""

    def __init__(
        self,
        parent: ViewerWindow | None,
        *,
        summary: str,
        dump_path: Path,
        dump_text: str,
        from_last_session: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle(
            "Crash report from your last session" if from_last_session else "Crash report"
        )
        self.resize(700, 500)
        self._dump_path = dump_path

        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        path_label = QLabel(f"Saved to: {dump_path}")
        path_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(summary_label)
        layout.addWidget(path_label)
        if not from_last_session:
            warning_label = QLabel(
                "The application may now be in an inconsistent state. If you "
                "have unsaved work, save it to a new file before continuing."
            )
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Monospace"))
        self.text.setPlainText(dump_text)
        layout.addWidget(self.text, stretch=1)

        copy_btn = QPushButton("Copy to clipboard")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        open_folder_btn = QPushButton("Open containing folder")
        open_folder_btn.clicked.connect(self._open_containing_folder)
        btn_row = QHBoxLayout()
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(open_folder_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        if from_last_session:
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(self.close)
            buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)
        else:
            buttons = QDialogButtonBox()
            save_as_btn = buttons.addButton("Save As...", QDialogButtonBox.ActionRole)
            save_as_btn.clicked.connect(self._save_as)
            # Continue is the default, not Quit: the app survives an
            # unhandled exception in this PyQt5 (confirmed by probe), and
            # forcing a quit here would destroy unsaved edits for no reason.
            continue_btn = buttons.addButton("Continue", QDialogButtonBox.AcceptRole)
            continue_btn.setDefault(True)
            quit_btn = buttons.addButton("Quit", QDialogButtonBox.DestructiveRole)
            quit_btn.clicked.connect(self._quit)
            buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _copy_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.text.toPlainText())

    def _open_containing_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._dump_path.parent)))

    def _save_as(self) -> None:
        parent = self.parent()
        if parent is not None and hasattr(parent, "save_as"):
            parent.save_as()

    def _quit(self) -> None:
        QApplication.instance().quit()
