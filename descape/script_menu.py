"""Tools > Scripts: the folder-backed submenu, consent, and a run's dialogs.

The submenu holds Open Scripts Folder, a rebindable command that creates the
folder and opens it. The script rows above it are reread from the folder each
time the submenu opens (`aboutToShow`), so a file dropped in shows up without
a restart. With addons switched off for the session (`--no-addons`,
`DESCAPE_NO_ADDONS`) every row is disabled and says why.

Running a row: the window's refusals first, then the file's bytes are read
once and hashed. A (resolved path, digest) pair the user has not agreed to
asks "Run script?" (No is the default); Yes stores the pair, so the next run
of the same bytes asks nothing. The bytes hashed are the bytes run. The run
itself is `ViewerWindow.run_code()`, which Tools > Addons shares.

A run's progress dialog appears only once the code first calls
`editor.progress()`. It is application-modal, since the history's undo does
not check the open stroke: no menu, key or click reaches the window while it
pumps events. Output goes to Help > Debug Log, an error to a message box with
the traceback behind Show Details.

Kept out of the viewer family: this module is Qt, but owns no window state.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QAction, QApplication, QMenu, QMessageBox, QProgressDialog

from descape import debug_log, legacy_write, scripting, settings

# How much of a run's printed output Help > Debug Log keeps: the last this many characters.
OUTPUT_LOG_LIMIT = 8000
# How often a script's editor.progress() repaints its dialog; a cancel is seen on every call.
PROGRESS_REFRESH_S = 0.05

_NO_SCRIPTS_TEXT = "(No scripts)"
_NO_SCRIPTS_TIP = "Put .py files in the scripts folder: Open Scripts Folder creates and opens it"


def build(window, tools_menu: QMenu) -> QMenu:
    """Tools > Scripts, with Open Scripts Folder. Sets `window.scripts_menu`
    and `window.open_scripts_folder_action`; the rows come on aboutToShow."""
    menu = tools_menu.addMenu("&Scripts")
    menu.menuAction().setToolTip("Run a Python script from your scripts folder against the open scenario")
    menu.setToolTipsVisible(True)
    action = QAction("&Open Scripts Folder", window)
    action.setToolTip("Creates your scripts folder if needed and opens it in your file manager")
    action.triggered.connect(lambda: open_scripts_folder(window))
    menu.addAction(action)
    menu.aboutToShow.connect(lambda: refresh_rows(window))
    window.scripts_menu = menu
    window.open_scripts_folder_action = action
    return menu


def disabled_reason() -> str:
    """Why scripts cannot run this session, or ""."""
    switch = settings.addons_disabled_by(sys.argv, os.environ)
    return f"Scripts are off for this session ({switch})" if switch else ""


def refresh_rows(window) -> None:
    """Rereads the scripts folder into the rows above Open Scripts Folder."""
    menu = window.scripts_menu
    for action in menu.actions():
        if action is window.open_scripts_folder_action:
            break
        menu.removeAction(action)
        if not action.isSeparator():
            action.deleteLater()
    entries = scripting.discover_scripts(scripting.script_root())
    off = disabled_reason()
    rows: list[QAction] = []
    for entry in entries:
        row = QAction(entry.title, window)
        row.setToolTip(off or str(entry.path))
        row.setEnabled(not off)
        row.triggered.connect(lambda _checked=False, e=entry: run_entry(window, e))
        rows.append(row)
    if not rows:
        row = QAction(_NO_SCRIPTS_TEXT, window)
        row.setToolTip(_NO_SCRIPTS_TIP)
        row.setEnabled(False)
        rows.append(row)
    first = window.open_scripts_folder_action
    for row in rows:
        menu.insertAction(first, row)
    menu.insertSeparator(first)


def open_scripts_folder(window) -> None:
    root = scripting.script_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        window._log_status(f"Couldn't create the scripts folder {root}: {error}", level=debug_log.WARNING)
        return
    if open_folder(root):
        window._log_status(f"Opened the scripts folder: {root}")
    else:
        window._log_status(f"Couldn't open a file manager. The scripts folder is {root}", level=debug_log.WARNING)


def open_folder(path: Path) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def refusal(window) -> str | None:
    """Why no code may run against the window's document right now, or None."""
    off = disabled_reason()
    if off:
        return off
    if window.scenario is None:
        return "open a scenario first"
    # A pre-DE document runs code once its band saves any family back; the run checks what it wrote (TASK-520).
    family = legacy_write.ANY if window._file_is_legacy() else None
    reason = window.game_data_lock_reason(family)
    if reason is not None:
        return reason
    if window._busy:
        return "the window is busy"
    if window._stroke_in_progress():
        return "an edit is still in progress"
    return None


def confirm_run(window, path: Path) -> bool:
    """The "Run script?" box. No is the default: Enter on a surprise box must not run code."""
    box = QMessageBox(window)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Run script?")
    box.setText(f"Run this script?\n\n{path}")
    box.setInformativeText(
        "A script is ordinary Python with your full privileges, not sandboxed. Run only code you trust. "
        "You won't be asked again until the file changes."
    )
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    box.setDefaultButton(QMessageBox.No)
    return box.exec_() == QMessageBox.Yes


def run_entry(window, entry: scripting.ScriptEntry) -> scripting.ScriptOutcome | None:
    """Runs one script row: refusals, consent, then the run. None when it never started."""
    label = f'Script "{entry.title}"'
    reason = refusal(window)
    if reason is not None:
        window._log_status(f"{label} refused: {reason}", level=debug_log.WARNING)
        return None
    try:
        source = entry.path.read_bytes()
        key = str(entry.path.resolve())
    except OSError as error:
        window._log_status(f"{label} could not be read: {error}", level=debug_log.WARNING)
        return None
    digest = scripting.content_digest(source)
    if (key, digest) not in settings.get_confirmed_scripts():
        if not confirm_run(window, entry.path):
            window._log_status(f"{label} not run")
            return None
        settings.add_confirmed_script(key, digest)
    # The box pumped events: an edit may have started behind it.
    reason = refusal(window)
    if reason is not None:
        window._log_status(f"{label} refused: {reason}", level=debug_log.WARNING)
        return None
    # As Save does: a value still typed in a field lands before the script reads the map, not after.
    window._commit_focused_field()
    return window.run_code(entry.title, label, lambda run: scripting.run_script(entry, source, run))


class _RunDialog(QProgressDialog):
    """A progress dialog that stays shown when cancelled. Hiding it would end
    its modal block inside the same event pump, letting a queued key or click
    reach the window while the run still holds the stroke."""

    def __init__(self, parent):
        super().__init__("", "Cancel", 0, 1, parent)
        self.cancel_requested = False
        self.canceled.disconnect(self.cancel)
        self.canceled.connect(self._request_cancel)

    def _request_cancel(self) -> None:
        self.cancel_requested = True
        self.setLabelText("Cancelling...")

    def reject(self) -> None:
        self._request_cancel()

    def closeEvent(self, event) -> None:
        event.ignore()
        self._request_cancel()


class Progress:
    """editor.progress()'s window side: a dialog made on the first call,
    repainted at most every PROGRESS_REFRESH_S, raising Cancelled once its
    Cancel is pressed."""

    def __init__(self, window, title: str):
        self._window = window
        self._title = title
        self.dialog: _RunDialog | None = None
        self._last = 0.0

    def __call__(self, done: int, total: int, note: str) -> None:
        now = time.perf_counter()
        if self.dialog is None:
            dialog = _RunDialog(self._window)
            dialog.setWindowTitle(self._title)
            dialog.setWindowModality(Qt.ApplicationModal)
            dialog.setAutoReset(False)
            dialog.setAutoClose(False)
            dialog.setMinimumDuration(0)
            self.dialog = dialog
            self._last = 0.0
        dialog = self.dialog
        if now - self._last >= PROGRESS_REFRESH_S or done >= total:
            self._last = now
            total = max(1, int(total))
            dialog.setMaximum(total)
            dialog.setLabelText(note or f"{max(0, int(done)):,} of {total:,}")
            dialog.setValue(min(max(0, int(done)), total))
            dialog.show()
            QApplication.processEvents()
        if dialog.cancel_requested:
            raise scripting.Cancelled()

    def close(self) -> None:
        if self.dialog is not None:
            self.dialog.hide()
            self.dialog.deleteLater()
            self.dialog = None


def report(window, outcome: scripting.ScriptOutcome) -> None:
    """The status line, the Debug Log, and for an error the message box."""
    short, lines = scripting.summarise(outcome)
    # A failed check's first finding is already the error in `short`.
    lines = [line for line in lines if line not in short]
    failed = bool(outcome.error) or outcome.incomplete
    level = debug_log.WARNING if failed else debug_log.INFO
    window._log_status(short, level=level)
    for line in lines:
        window._log_status(f"  {line}", level=level)
    output = outcome.output
    if output:
        kept = output[-OUTPUT_LOG_LIMIT:]
        cut = f" (last {OUTPUT_LOG_LIMIT:,} of {len(output):,} characters)" if len(kept) < len(output) else ""
        debug_log.log(f"{outcome.label} output{cut}:\n{kept.rstrip()}")
    if outcome.detail:
        debug_log.log(f"{outcome.label} traceback:\n{outcome.detail}", level=level)
    if outcome.incomplete or (outcome.error and not outcome.cancelled):
        details = "\n\n".join(part for part in (outcome.detail, _output_tail(output)) if part)
        show_error(window, f"{outcome.label} failed", "\n".join([short, *lines]), details)


def _output_tail(output: str) -> str:
    return f"Output:\n{output[-OUTPUT_LOG_LIMIT:].rstrip()}" if output.strip() else ""


def show_error(window, title: str, text: str, details: str) -> None:
    box = QMessageBox(window)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    if details:
        box.setDetailedText(details)
    box.exec_()


__all__ = [
    "OUTPUT_LOG_LIMIT",
    "Progress",
    "build",
    "confirm_run",
    "disabled_reason",
    "open_folder",
    "open_scripts_folder",
    "refresh_rows",
    "refusal",
    "report",
    "run_entry",
    "show_error",
]
