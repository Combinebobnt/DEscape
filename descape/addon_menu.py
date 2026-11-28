"""Tools > Addons: loading addons at startup, their command rows, and
running one.

The submenu holds Open Addons Folder, a rebindable command that creates the
folder and opens it. Each time the submenu opens (`aboutToShow`) the rows
above it are rebuilt from `addon_api.ADDON_COMMANDS`, each command's
`refusal` evaluated then, and a disabled summary row below it says what
loading did ("2 loaded, 1 failed: see Help > Debug Log").

`load_startup()` runs once, queued after the window is shown, so the "Run
addon?" box never comes before the launch's ready signal. A command runs
through `ViewerWindow.run_code()`, the transaction Tools > Scripts uses.

Kept out of the viewer family: this module is Qt, but owns no window state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import QAction, QMenu, QMessageBox

from descape import addon_api, addons, debug_log, edit_history, script_menu, scripting, settings

_NO_COMMANDS_TEXT = "(No addon commands)"
_NO_COMMANDS_TIP = "Put addon .py files or packages in the addons folder: Open Addons Folder creates and opens it"

_loaded = False


def build(window, tools_menu: QMenu) -> QMenu:
    """Tools > Addons, with Open Addons Folder. Sets `window.addons_menu` and
    `window.open_addons_folder_action`; the rows come on aboutToShow."""
    menu = tools_menu.addMenu("A&ddons")
    menu.menuAction().setToolTip("Commands added by addons in your addons folder")
    menu.setToolTipsVisible(True)
    action = QAction("&Open Addons Folder", window)
    action.setToolTip("Creates your addons folder if needed and opens it in your file manager")
    action.triggered.connect(lambda: open_addons_folder(window))
    menu.addAction(action)
    menu.aboutToShow.connect(lambda: refresh_rows(window))
    window.addons_menu = menu
    window.open_addons_folder_action = action
    return menu


def _probe_editor(window) -> scripting.Editor | None:
    """An `editor` over the open document for a refusal to read; its writes refuse."""
    if window.scenario is None:
        return None
    run = scripting.ScriptRun(
        window.scenario,
        window.edit_history,
        selected_units=lambda: [entry.unit for entry in window._selected_unit_entries()],
        selected_tiles=lambda: (),
    )
    return run.editor


def _tile_states(loaded) -> list:
    return [edit_history.tile_state(tile) for tile in loaded.map_manager.terrain]


def _restore_tiles(loaded, states: list) -> bool:
    changed = False
    for tile, state in zip(loaded.map_manager.terrain, states, strict=True):
        now = edit_history.tile_state(tile)
        if now != state or [*map(type, now)] != [*map(type, state)]:
            tile.terrain_id, tile.elevation, tile.layer = state
            changed = True
    return changed


def command_refusal(window, command: addon_api.AddonCommand) -> str | None:
    """Why `command` cannot run now: the window's refusals, then its own.
    A refusal only reads: one that changed a unit or tile is put back and
    greys its row, since nothing outside a run is undone or checked."""
    reason = script_menu.refusal(window)
    if reason is not None or command.refusal is None:
        return reason
    loaded = window.scenario
    image, tiles = scripting.capture_image(loaded), _tile_states(loaded)
    try:
        reason = command.refusal(_probe_editor(window))
        reason = str(reason) if reason else None
    except (Exception, SystemExit) as error:  # noqa: BLE001 -- addon code: a raise greys the row, never the menu
        debug_log.log(f'Addon "{command.title}" refusal raised:\n{scripting.detail(error)}', level=debug_log.WARNING)
        reason = f"its availability check failed: {scripting.describe(error)}"
    finally:
        touched_units = scripting.restore_image(image, loaded)
        touched_tiles = _restore_tiles(loaded, tiles)
    if touched_units or touched_tiles:
        debug_log.log(f'Addon "{command.title}" refusal changed the scenario; put back', level=debug_log.WARNING)
        return "its availability check changed the scenario"
    return reason


def refresh_rows(window) -> None:
    """Rebuilds the command rows above Open Addons Folder and the summary row below it."""
    menu = window.addons_menu
    folder = window.open_addons_folder_action
    for action in menu.actions():
        if action is folder:
            continue
        menu.removeAction(action)
        if not action.isSeparator():
            action.deleteLater()
    rows: list[QAction] = []
    for command in addon_api.ADDON_COMMANDS.entries().values():
        try:
            reason = command_refusal(window, command)
        except Exception as error:  # noqa: BLE001 -- one bad entry must not empty the menu
            debug_log.log(f"Addon command {command!r} row failed:\n{scripting.detail(error)}", level=debug_log.WARNING)
            continue
        row = QAction(command.title, window)
        tip = command.status_tip or command.title
        row.setToolTip(f"{tip} (unavailable: {reason})" if reason else tip)
        row.setEnabled(reason is None)
        row.triggered.connect(lambda _checked=False, c=command: run_command(window, c))
        rows.append(row)
    if not rows:
        row = QAction(_NO_COMMANDS_TEXT, window)
        row.setToolTip(_NO_COMMANDS_TIP)
        row.setEnabled(False)
        rows.append(row)
    for row in rows:
        menu.insertAction(folder, row)
    menu.insertSeparator(folder)
    summary = QAction(addons.summary_text(addons.last_report()), window)
    summary.setToolTip("What happened to each addon at startup is in Help > Debug Log")
    summary.setEnabled(False)
    menu.addAction(summary)


def open_addons_folder(window) -> None:
    root = addons.addon_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        window._log_status(f"Couldn't create the addons folder {root}: {error}", level=debug_log.WARNING)
        return
    if script_menu.open_folder(root):
        window._log_status(f"Opened the addons folder: {root}")
    else:
        window._log_status(f"Couldn't open a file manager. The addons folder is {root}", level=debug_log.WARNING)


def confirm_load(window, path: Path) -> bool:
    """The "Run addon?" box. No is the default: Enter on a surprise box must not run code."""
    box = QMessageBox(window)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Run addon?")
    box.setText(f"Load this addon?\n\n{path}")
    box.setInformativeText(
        "An addon is ordinary Python with your full privileges, not sandboxed, and runs each time DEscape "
        "starts. Load only code you trust. You won't be asked again until it changes."
    )
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    box.setDefaultButton(QMessageBox.No)
    return box.exec_() == QMessageBox.Yes


def load_startup(window) -> addons.AddonReport | None:
    """Loads the addons folder once per process, asking about each new or
    changed addon (each Yes stored as it is given), and logs the report."""
    global _loaded
    if _loaded:
        return addons.last_report()
    _loaded = True

    def ask(path: Path, digest: str) -> bool:
        if not confirm_load(window, path):
            return False
        settings.add_confirmed_addon(str(path), digest)
        return True

    report = addons.load_addons(
        [addon_api.ADDON_COMMANDS], argv=sys.argv, env=os.environ, ask=ask, confirmed=settings.get_confirmed_addons()
    )
    failed = bool(report.failed)
    for line in addons.report_lines(report):
        debug_log.log(line, level=debug_log.WARNING if failed else debug_log.INFO)
    if report.records:
        window._log_status(f"Addons: {addons.summary_text(report)}", level=debug_log.WARNING if failed else debug_log.INFO)
    return report


def run_command(window, command: addon_api.AddonCommand) -> scripting.ScriptOutcome | None:
    """Runs one command row: the refusals, then the run. None when it never started."""
    label = f'Addon "{command.title}"'
    reason = command_refusal(window, command)
    if reason is not None:
        window._log_status(f"{label} refused: {reason}", level=debug_log.WARNING)
        return None
    window._commit_focused_field()
    return window.run_code(command.title, label, lambda run: scripting.call(command.run, run, title=command.title, label=label))


__all__ = [
    "build",
    "command_refusal",
    "confirm_load",
    "load_startup",
    "open_addons_folder",
    "refresh_rows",
    "run_command",
]
