"""File > Data Mod: the window's half of GH #58's data mods (descape/data_mods.py).

The menu lists "None (the install's data)" and every discovered data mod.
Picking one puts its derived tables in effect through game_dataset's overlay:
at once on a cache hit, else after a derive subprocess (a QProcess running
data_mods.derive_command(), since a parse holds the GIL for 5-18 s), with a
persistent "Parsing <mod>…" label until it finishes. A finished parse swaps
only once the window is idle: refresh_map() silently no-ops while the
window is busy, and a swap mid-stroke would redraw half a stroke with other
tables, so the swap re-arms a short timer until neither holds.

Per-scenario memory (settings.get_data_mod_for()): a saved document's choice
is written on a pick and by Save As, and read when the document opens
(for_document(), before its first render). An untitled document never reads
or writes it: File > New keeps the session's choice, and a recovered autosave
carries its own in the autosave row. A mod that is gone, unreadable this
session or uncached without genieutils-py falls back to the install's data
with one log line, and its key stays until the next pick or Save As.

Under Return of Rome data the overlay is inert (game_dataset), so the menu's
mods are disabled and nothing is remembered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from PyQt5.QtCore import QObject, QProcess, QTimer
from PyQt5.QtWidgets import QAction, QActionGroup, QLabel, QMenu

from descape import asset_source, data_mods, game_dataset, settings

# How often a finished parse re-checks for an idle window before swapping.
APPLY_RETRY_MS = 200
# load_scenario()'s default: a saved document's remembered mod, the session's for an untitled one.
FROM_PATH = object()


@dataclass
class _Job:
    """The running derive: its mod, cache key and process, and whether a pick started it."""

    mod: data_mods.DataMod
    key: str
    process: QProcess
    from_pick: bool
    started: float


class DataModMenu(QObject):
    """One per window. `window` is the ViewerWindow, read for its busy state,
    open document and log, and refreshed after a swap."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        # What the user or the open document asked for; in effect once a parse finishes.
        self._wanted: data_mods.DataMod | None = None
        # What game_dataset has in effect, as a DataMod.
        self._effective: data_mods.DataMod | None = None
        # status_path -> why it cannot be used, for this session (a failed parse).
        self._failed: dict[str, str] = {}
        self._job: _Job | None = None
        # A finished parse's swap, waiting for an idle window: (tables, mod).
        self._pending: tuple[game_dataset.DataModTables, data_mods.DataMod] | None = None
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(APPLY_RETRY_MS)
        self._apply_timer.timeout.connect(self._apply_pending)
        self.parsing_label = QLabel()
        self.parsing_label.setVisible(False)
        self.menu: QMenu | None = None
        self._group: QActionGroup | None = None

    # -- the menu -----------------------------------------------------------

    def build(self, parent_menu: QMenu) -> QMenu:
        """Adds File > Data Mod to `parent_menu`; its items are rebuilt each time it opens."""
        self.menu = parent_menu.addMenu("Data &Mod")
        self.menu.menuAction().setToolTip(
            "Show the open scenario with a data mod's units and stats, remembered per scenario"
        )
        self.menu.setToolTipsVisible(True)
        self.menu.aboutToShow.connect(self.rebuild)
        return self.menu

    def rebuild(self) -> None:
        menu = self.menu
        menu.clear()
        if self._group is not None:
            self._group.deleteLater()
        group = self._group = QActionGroup(menu)
        none_action = QAction("None (the game's own data)", menu, checkable=True)
        none_action.setChecked(self._wanted is None)
        none_action.triggered.connect(lambda checked=False: self.choose(None))
        group.addAction(none_action)
        menu.addAction(none_action)
        menu.addSeparator()
        mods = data_mods.discover()
        if not mods:
            empty = menu.addAction("No data mods found")
            empty.setEnabled(False)
            empty.setToolTip("A data mod ships resources/_common/dat/empires2_x2_p1.dat in your profile's mods folder")
        for mod in mods:
            action = QAction(mod.title, menu, checkable=True)
            action.setChecked(mod == self._wanted)
            reason = self.unusable_reason(mod)
            action.setEnabled(reason is None)
            action.setToolTip(reason or mod.status_path)
            action.triggered.connect(lambda checked=False, mod=mod: self.choose(mod))
            group.addAction(action)
            menu.addAction(action)

    def unusable_reason(self, mod: data_mods.DataMod) -> str | None:
        """Why `mod` cannot be picked right now, or None."""
        if game_dataset.is_ror():
            return "Data mods change Age of Empires II data; this document shows Return of Rome's"
        if mod.status_path in self._failed:
            return self._failed[mod.status_path]
        install = asset_source.get_install_path()
        if install is None:
            return "Set the game's install folder in Settings first: a data mod's tables are read against it"
        if not data_mods.genieutils_available() and self._cached(mod, install) is None:
            return "Reading this mod's .dat needs genieutils-py, which is not installed"
        return None

    # -- choosing -----------------------------------------------------------

    def choose(self, mod: data_mods.DataMod | None) -> None:
        """A menu pick: puts `mod` in effect (or starts parsing it) and remembers it for a saved document."""
        if mod is not None:
            reason = self.unusable_reason(mod)
            if reason is not None:
                self._window._log_status(f"Data mod {mod.title} cannot be used: {reason}")
                return
        self._wanted = mod
        self.remember_current()
        if mod is None:
            self._stop_job()
            self._swap(None, None)
            return
        install = asset_source.get_install_path()
        tables = self._cached(mod, install)
        if tables is not None:
            self._stop_job()
            self._swap(tables, mod)
        elif self._job is None or self._job.mod != mod:
            self._start_job(mod, install, from_pick=True)

    def for_document(self, path, *, untitled: bool, remembered=FROM_PATH) -> bool:
        """The opening document's data mod, set before its first render (load_scenario(),
        inside its busy section, so this never defers). `remembered` is FROM_PATH, or a
        recovered autosave's identity (None for the install's data). True when the
        tables in effect changed; a miss shows the install's data while it parses."""
        if remembered is FROM_PATH:
            if untitled:
                return False  # File > New and a resized untitled map keep the session's choice
            remembered = settings.get_data_mod_for(path)
        self._pending = None
        self._apply_timer.stop()
        mod = None if remembered is None or game_dataset.is_ror() else self._resolve(remembered)
        install = asset_source.get_install_path()
        tables = None if mod is None else self._cached(mod, install)
        # Before any job starts: a job that fails at once falls back to what is in effect.
        changed = self._set_effective(tables, mod if tables is not None else None)
        self._wanted = mod
        if self._job is not None and self._job.mod != mod:
            self._stop_job()
        elif self._job is not None:
            self._job.from_pick = False  # adopted: a failure must not rewrite this document's memory
        if mod is not None and tables is None and self._job is None:
            self._start_job(mod, install, from_pick=False)
        return changed

    def _resolve(self, remembered: dict) -> data_mods.DataMod | None:
        """The discovered mod `remembered` names, if it can be used; else None, said once in the log."""
        title = remembered.get("title") or remembered.get("status_path")
        mod = data_mods.find(remembered, data_mods.discover())
        if mod is None:
            self._window._log_status(f"Data mod {title} was not found; showing the game's own data")
            return None
        reason = self.unusable_reason(mod)
        if reason is not None:
            self._window._log_status(f"Data mod {mod.title} cannot be used: {reason}; showing the game's own data")
            return None
        return mod

    def identity(self) -> dict | None:
        """The chosen mod as settings and the autosave index store it; None for the
        install's data and always under Return of Rome, where a mod is inert."""
        if self._wanted is None or game_dataset.is_ror():
            return None
        return data_mods.identity(self._wanted)

    def remember_current(self) -> None:
        """The chosen mod under the open saved document's path: a pick, Save As and a first save."""
        window = self._window
        if window.scenario is None or window._untitled:
            return
        settings.set_data_mod_for(window.scenario.path, self.identity())

    def title_suffix(self) -> str:
        """The window title's " [mod]" while a data mod's tables are in effect."""
        mod = game_dataset.data_mod()
        return "" if mod is None else f" [{mod.title}]"

    def install_changed(self) -> None:
        """Settings' install folder moved: the chosen mod's tables were read against
        the old one, so it is picked again (a new cache key, likely a parse)."""
        wanted = self._wanted
        if wanted is None:
            return
        self._failed.pop(wanted.status_path, None)
        self._stop_job()
        self.choose(wanted)

    def _cached(self, mod: data_mods.DataMod, install) -> game_dataset.DataModTables | None:
        try:
            return data_mods.tables_for(mod, data_mods.entry_key(mod, install))
        except OSError as exc:  # the mod's .dat vanished or cannot be read
            self._failed[mod.status_path] = f"its .dat cannot be read ({exc})"
            return None

    # -- the derive subprocess ----------------------------------------------

    def _start_job(self, mod: data_mods.DataMod, install, *, from_pick: bool) -> None:
        self._stop_job()
        # An earlier pick's swap still waiting for idle would otherwise land during this parse.
        self._pending = None
        self._apply_timer.stop()
        try:
            key = data_mods.entry_key(mod, install)
        except OSError as exc:
            self._fail(mod, f"its .dat cannot be read ({exc})", from_pick)
            return
        argv, cwd = data_mods.derive_command(mod, install, key)
        process = QProcess(self)
        process.setWorkingDirectory(str(cwd))
        # The generators' progress lines; only stderr's last line (the failure) is read.
        process.setStandardOutputFile(QProcess.nullDevice())
        job = _Job(mod, key, process, from_pick, time.perf_counter())
        process.finished.connect(lambda code, status, job=job: self._on_finished(job, code, status))
        process.errorOccurred.connect(lambda error, job=job: self._on_error(job, error))
        self._job = job
        self.parsing_label.setText(f"Parsing {mod.title}…")
        self.parsing_label.setVisible(True)
        self._window._log_status(f"Parsing data mod {mod.title} ({mod.dat}); this takes 5-25 s")
        process.start(argv[0], argv[1:])

    def _stop_job(self) -> None:
        job, self._job = self._job, None
        self.parsing_label.setVisible(False)
        if job is not None and job.process.state() != QProcess.NotRunning:
            job.process.kill()

    def shutdown(self) -> None:
        """Window close: no parse outlives the window, and no swap lands after it."""
        job = self._job
        self._stop_job()
        self._pending = None
        self._apply_timer.stop()
        if job is not None:
            job.process.waitForFinished(2000)

    def _on_error(self, job: _Job, error) -> None:
        # FailedToStart emits no finished(); every other error is followed by one.
        if error == QProcess.FailedToStart and job is self._job:
            self._job = None
            self.parsing_label.setVisible(False)
            self._fail(job.mod, f"the parser could not start ({job.process.errorString()})", job.from_pick)
            job.process.deleteLater()

    def _on_finished(self, job: _Job, code: int, status) -> None:
        job.process.deleteLater()
        if job is not self._job:
            return  # stopped or replaced
        self._job = None
        self.parsing_label.setVisible(False)
        stderr = bytes(job.process.readAllStandardError()).decode("utf-8", "replace")
        tables = data_mods.tables_for(job.mod, job.key) if status == QProcess.NormalExit and code == 0 else None
        if tables is None:
            lines = [line.strip() for line in stderr.splitlines() if line.strip()]
            if status != QProcess.NormalExit:
                message = "the parser crashed"
            else:
                message = lines[-1] if lines else f"the parser exited with code {code}"
            self._fail(job.mod, message, job.from_pick)
            return
        self._window._log_status(f"Parsed data mod {job.mod.title} in {time.perf_counter() - job.started:.1f} s")
        if job.mod == self._wanted:
            self._swap(tables, job.mod)

    def _fail(self, mod: data_mods.DataMod, message: str, from_pick: bool) -> None:
        """A mod that cannot be read: kept out of the menu this session, the current data kept."""
        self._failed[mod.status_path] = message
        self._window._log_status(f"Data mod {mod.title} cannot be used: {message}")
        if mod == self._wanted:
            self._wanted = self._effective
            if from_pick:
                self.remember_current()

    # -- swapping -----------------------------------------------------------

    def _swap(self, tables: game_dataset.DataModTables | None, mod: data_mods.DataMod | None) -> None:
        """Puts `tables` in effect once the window is idle, then refreshes everything built from them."""
        self._pending = (tables, mod)
        self._apply_pending()

    def _apply_pending(self) -> None:
        if self._pending is None:
            return
        window = self._window
        if window._busy or window._stroke_in_progress():
            self._apply_timer.start()
            return
        (tables, mod), self._pending = self._pending, None
        if self._set_effective(tables, mod):
            window._refresh_game_assets()
            window._after_game_data_change()

    def _set_effective(self, tables: game_dataset.DataModTables | None, mod: data_mods.DataMod | None) -> bool:
        self._effective = mod
        changed = game_dataset.set_data_mod(tables)
        if changed:
            shown = "the game's own data" if game_dataset.data_mod() is None else f"data mod {mod.title}'s data"
            self._window._log_status(f"Showing {shown}")
        return changed
