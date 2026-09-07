"""App-chrome theming and the Help > Debug Log window.

_LIGHT_PALETTE moves with apply_theme() rather than being re-exported:
it is `global`-mutated there, so a re-export would not track it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
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
)

# DebugLogDialog's parent is annotated "ViewerWindow". `from __future__ import
# annotations` means that string is never evaluated, so this stays a
# type-checker-only import -- viewer.py imports this module, and a runtime
# import back would be a cycle.
if TYPE_CHECKING:
    from descape.viewer import ViewerWindow


# Cached the first time apply_theme() runs, before any dark palette is ever
# applied -- the baseline to restore when the user switches back to light.
# Captured from the Fusion style itself (style().standardPalette()), not
# whatever native platform style/palette was active before this app ever
# touched it: apply_theme() always forces Fusion (light or dark), so the
# "light" state should be Fusion's own default, not a different style's
# palette that would look inconsistent switching back and forth.
_LIGHT_PALETTE: QPalette | None = None


def _build_dark_palette() -> QPalette:
    """The standard Fusion dark palette recipe (Window/Base/Text/Button
    darkened, Highlight kept a legible blue, disabled-state colors dimmed
    separately so disabled controls don't just look identically dark) --
    no novel color choices here, this is the well-known combination most
    Fusion-dark Qt apps use."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(230, 230, 230))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))
    return palette


def apply_theme(app: QApplication, dark: bool) -> None:
    """App chrome only -- menus, dialogs, toolbars, the status bar. MapView's
    own colors (real per-tile terrain textures, unit dots, hover highlights)
    are untouched: they represent game data or are already dark, not UI
    styling that should shift with this toggle. Called once at startup
    (main(), from the persisted settings.get_dark_mode()) and live from the
    Appearance settings tab -- safe to call repeatedly, setStyle("Fusion")
    is idempotent and _LIGHT_PALETTE is only ever captured once."""
    global _LIGHT_PALETTE
    app.setStyle("Fusion")
    if _LIGHT_PALETTE is None:
        _LIGHT_PALETTE = app.style().standardPalette()
    app.setPalette(_build_dark_palette() if dark else _LIGHT_PALETTE)

class DebugLogDialog(QDialog):
    """Read-only viewer over debug_log's in-memory buffer. A snapshot at open
    time (and after Refresh), not a live tail -- simplest thing that's useful
    for "what did the app just do," not meant as a full log console."""

    def __init__(self, parent: "ViewerWindow"):
        super().__init__(parent)
        self.setWindowTitle("Debug Log")
        self.resize(640, 400)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
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
        parent: "ViewerWindow | None",
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
