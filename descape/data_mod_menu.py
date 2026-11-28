"""File > Data Mod: the window's half of GH #58's data mods (descape/data_mods.py).

The menu lists "None (the install's data)" and every discovered data mod.
Picking one puts its derived tables in effect through game_dataset's overlay:
at once on a cache hit, else after a derive subprocess (a QProcess running
data_mods.derive_command(), since a parse holds the GIL for 5-18 s), with the
status bar's progress bar following its steps until it finishes (GH #359).
A finished parse swaps only once the window is idle: refresh_map() silently
no-ops while the window is busy, and a swap mid-stroke would redraw half a
stroke with other tables, so the swap re-arms a short timer until neither holds.

Per-scenario memory (settings.get_data_mod_for()): a saved document's choice
is written on a pick and by Save As, and read when the document opens
(for_document(), before its first render). An untitled document never reads
or writes it: File > New keeps the session's choice, and a recovered autosave
carries its own in the autosave row. A mod that is gone, unreadable this
session or uncached without genieutils-py falls back to the install's data
with one log line, and its key stays until the next pick or Save As.

Under Return of Rome data the overlay is inert (game_dataset), so the menu's
mods are disabled and nothing is remembered.

A mod whose derive simplified objects or rows (GH #262, dat_derive's lenient
mode) says so once per window and cache key, in the status log when its tables
take effect, and from then on in its menu item's tooltip. The status log words
it for the user (GH #323): one line per warning kind, in WARNING_TEXT's words,
with its count of distinct objects or terrains and a few example names, and
the meta.json path that holds the generators' own text for a bug report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from PyQt5 import sip
from PyQt5.QtCore import QObject, QProcess, QTimer
from PyQt5.QtWidgets import QAction, QActionGroup, QMenu

from descape import (
    asset_source,
    dat_cache,
    data_mods,
    debug_log,
    game_dataset,
    games,
    mod_graphics,
    object_catalog,
    settings,
    unit_sprites,
)

# How often a finished parse re-checks for an idle window before swapping.
APPLY_RETRY_MS = 200
# How many of a mod's missing art files the status log lists (art_summary()).
WARNING_LINES = 20
# How many example names a warning kind's line gives.
WARNING_EXAMPLES = 3
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
    # Stdout read so far past its last newline.
    partial: bytes = field(default=b"")


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
        # Cache keys whose derive warnings this window's status log has shown.
        self._warned: set[str] = set()
        # status_path -> the warning count of the summary last logged for it.
        self._warning_counts: dict[str, int] = {}
        # A finished parse's swap, waiting for an idle window: (tables, mod).
        self._pending: tuple[game_dataset.DataModTables, data_mods.DataMod] | None = None
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(APPLY_RETRY_MS)
        self._apply_timer.timeout.connect(self._apply_pending)
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
            action.setToolTip(reason or self._item_tooltip(mod))
            action.triggered.connect(lambda checked=False, mod=mod: self.choose(mod))
            group.addAction(action)
            menu.addAction(action)

    def _item_tooltip(self, mod: data_mods.DataMod) -> str:
        """A usable mod's tooltip: its status path, plus how much it simplifies once the status log has the summary."""
        count = self._warning_counts.get(mod.status_path)
        if not count:
            return mod.status_path
        return f"{mod.status_path} ({_object_count(count)} shown simplified; see the status log)"

    def unusable_reason(self, mod: data_mods.DataMod) -> str | None:
        """Why `mod` cannot be picked right now, or None."""
        shown = game_dataset.game()
        if not shown.data_mods:
            takes = " or ".join(game.display_name for game in games.GAMES if game.data_mods)
            return f"Data mods change {takes} data; this document shows {shown.display_name}'s"
        if mod.status_path in self._failed:
            return self._failed[mod.status_path]
        install = asset_source.get_de_install_path()
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
                self._window._log_status(f"Data mod {mod.title} cannot be used: {reason}", level=debug_log.WARNING)
                return
        self._wanted = mod
        self.remember_current()
        if mod is None:
            self._stop_job()
            self._swap(None, None)
            return
        install = asset_source.get_de_install_path()
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
        mod = None if remembered is None or not game_dataset.game().data_mods else self._resolve(remembered)
        install = asset_source.get_de_install_path()
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
            self._window._log_status(f"Data mod {title} was not found; showing the game's own data", level=debug_log.WARNING)
            return None
        reason = self.unusable_reason(mod)
        if reason is not None:
            self._window._log_status(f"Data mod {mod.title} cannot be used: {reason}; showing the game's own data", level=debug_log.WARNING)
            return None
        return mod

    def identity(self) -> dict | None:
        """The chosen mod as settings and the autosave index store it; None for the
        install's data and always under a game without data mods (Return of Rome), where a mod is inert."""
        if self._wanted is None or not game_dataset.game().data_mods:
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
        job = _Job(mod, key, process, from_pick, time.perf_counter())
        # Stdout carries dat_derive's progress lines; of stderr only the last line (the failure) is read.
        process.readyReadStandardOutput.connect(lambda job=job: self._on_output(job))
        process.finished.connect(lambda code, status, job=job: self._on_finished(job, code, status))
        process.errorOccurred.connect(lambda error, job=job: self._on_error(job, error))
        self._job = job
        self._window.load_progress.start("data_mod", f"Parsing {mod.title}: starting", 0)
        self._window._log_status(f"Parsing data mod {mod.title} ({mod.dat}); this takes 5-25 s")
        process.start(argv[0], argv[1:])

    def _stop_job(self) -> None:
        job, self._job = self._job, None
        self._window.load_progress.finish("data_mod")
        if job is not None and job.process.state() != QProcess.NotRunning:
            job.process.kill()

    def shutdown(self) -> None:
        """Window close: no parse outlives the window, and no swap lands after it."""
        self._stop_job()
        self._pending = None
        self._apply_timer.stop()
        # A replaced job too: killed but unseen, its QProcess would end inside its destructor (TASK-700).
        for process in self.findChildren(QProcess):
            if process.state() != QProcess.NotRunning:
                process.kill()
                process.waitForFinished(2000)

    def _on_error(self, job: _Job, error) -> None:
        # FailedToStart emits no finished(); every other error is followed by one.
        if error == QProcess.FailedToStart and job is self._job:
            self._job = None
            self._window.load_progress.finish("data_mod")
            self._fail(job.mod, f"the parser could not start ({job.process.errorString()})", job.from_pick)
            job.process.deleteLater()

    def _on_output(self, job: _Job) -> None:
        if sip.isdeleted(job.process):
            return  # emitted by a still-running process's destructor: its window is going
        lines = (job.partial + bytes(job.process.readAllStandardOutput())).split(b"\n")
        job.partial = lines.pop()
        if job is not self._job:
            return  # stopped or replaced
        for line in lines:
            # The generators print their own lines here too; they parse to None.
            parsed = dat_cache.parse_progress_line(line.decode("utf-8", "replace"))
            if parsed is not None:
                percent, caption = parsed
                self._window.load_progress.step("data_mod", percent, f"Parsing {job.mod.title}: {caption}")

    def _on_finished(self, job: _Job, code: int, status) -> None:
        if sip.isdeleted(job.process):
            return  # as in _on_output(); a raise here is a qFatal abort
        job.process.deleteLater()
        if job is not self._job:
            return  # stopped or replaced
        self._job = None
        self._window.load_progress.finish("data_mod")
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
        self._window._log_status(f"Data mod {mod.title} cannot be used: {message}", level=debug_log.ERROR)
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
        if changed and tables is not None:
            if mod is not None:
                self._warning_counts[mod.status_path] = simplified_count(tables.warnings)
            if tables.warnings and tables.key not in self._warned:
                self._warned.add(tables.key)
                self._window._log_status(warning_summary(tables), level=debug_log.WARNING)
            if art := art_summary(tables):
                self._window._log_status(art)
        return changed


@dataclass(frozen=True)
class WarningText:
    """One warning kind's status log line, for a count of 1 (`one`) and of
    {n} (`many`). `subject` is what the kind's records name: "unit" (a unit
    const, which gets example names), "terrain" (a terrain id) or "none"."""

    subject: str
    one: str
    many: str


# Every tools/_game_root.WARNING_KINDS kind plus "other", dat_cache's fallback
# (tests/test_dat_derive.py pins both directions). What the user sees differently, and
# what to do about it where there is anything.
WARNING_TEXT: dict[str, WarningText] = {
    "parts_dropped": WarningText(
        "unit",
        "1 object is drawn without some of its parts (lights, flags, sails or annexes), or as a mark only",
        "{n} objects are drawn without some of their parts (lights, flags, sails or annexes), some as a mark only",
    ),
    "not_drawn": WarningText(
        "unit",
        "1 object is not drawn: only its mark shows",
        "{n} objects are not drawn: only their marks show",
    ),
    "no_garrison_flags": WarningText(
        "unit",
        "1 object shows no garrison flag when units are inside",
        "{n} objects show no garrison flag when units are inside",
    ),
    "age_art": WarningText(
        "unit",
        "1 building may show another age's look, or its annexes may",
        "{n} buildings may show another age's look, or their annexes may",
    ),
    "civ_art": WarningText(
        "unit",
        "1 building is drawn with the default look for some civilizations",
        "{n} buildings are drawn with the default look for some civilizations",
    ),
    "footprint": WarningText(
        "unit",
        "1 object takes up one tile for selection and placement, maybe less than in the game: leave room around it",
        "{n} objects take up one tile for selection and placement, maybe less than in the game: leave room around them",
    ),
    "placement_unchecked": WarningText(
        "none",
        "Some objects are placed without a terrain check: keep them on terrain the game allows",
        "Some objects are placed without a terrain check: keep them on terrain the game allows",
    ),
    "invisible_rule": WarningText(
        "none",
        "Some invisible objects may not count as invisible for Filters > Show Invisible Objects and the Eraser",
        "Some invisible objects may not count as invisible for Filters > Show Invisible Objects and the Eraser",
    ),
    "terrain_class": WarningText(
        "terrain",
        "1 terrain is not known as land, water or beach: the terrain list may file it elsewhere, "
        "and Auto beach lays no shoreline beside it",
        "{n} terrains are not known as land, water or beach: the terrain list may file them elsewhere, "
        "and Auto beach lays no shoreline beside them",
    ),
    "terrain_blend": WarningText(
        "terrain",
        "1 terrain is drawn without blending into its neighbours",
        "{n} terrains are drawn without blending into their neighbours",
    ),
    dat_cache.OTHER_KIND: WarningText(
        "none",
        "1 other object or table row is shown simplified",
        "{n} other objects or table rows are shown simplified",
    ),
}


def _object_count(count: int) -> str:
    return "1 object" if count == 1 else f"{count} objects"


def _file_count(count: int) -> str:
    return "1 art file" if count == 1 else f"{count} art files"


def art_summary(tables: game_dataset.DataModTables) -> str:
    """The status log's lines for the art a data mod's tables name that
    neither the install nor an active mod provides (GH #272), "" when none:
    a count, up to WARNING_LINES names, the rest's count. With Mod Art off,
    one line, since only Mod Art draws a data mod's art mod."""
    missing = [*unit_sprites.undrawable_stems(), *asset_source.undrawable_terrain_textures()]
    if not missing:
        return ""
    count = _file_count(len(missing))
    if not mod_graphics.enabled():
        return f"Data mod {tables.title}: {count} not in the game's install; View > Layers > Mod Art draws them from its art mod"
    lines = [
        f"Data mod {tables.title}: {count} that no active mod provides, drawn as marks or flat colours:",
        *(f"  {name}" for name in missing[:WARNING_LINES]),
    ]
    if len(missing) > WARNING_LINES:
        lines.append(f"  ... and {len(missing) - WARNING_LINES} more")
    return "\n".join(lines)


def _kind_of(warning) -> str:
    return warning.kind if warning.kind in WARNING_TEXT else dat_cache.OTHER_KIND


def _subjects(warnings) -> tuple[list[int], int]:
    """(the distinct ids `warnings` name, in order; how many name none). A tuple subject names each member."""
    ids: dict[int, None] = {}
    unnamed = 0
    for warning in warnings:
        subject = warning.subject
        if subject is None:
            unnamed += 1
        else:
            ids.update(dict.fromkeys(subject if isinstance(subject, tuple) else (subject,)))
    return list(ids), unnamed


def simplified_count(warnings) -> int:
    """How many objects or terrains `warnings` simplify: each distinct unit or
    terrain once across every kind, plus one per warning that names none."""
    named: set[tuple[str, int]] = set()
    unnamed = 0
    for warning in warnings:
        subject_type = WARNING_TEXT[_kind_of(warning)].subject
        ids, none = _subjects((warning,))
        named.update((subject_type, i) for i in ids)
        unnamed += none
    return len(named) + unnamed


def _kind_line(kind: str, warnings) -> tuple[int, str]:
    text = WARNING_TEXT[kind]
    ids, unnamed = _subjects(warnings)
    count = len(ids) + unnamed
    line = (text.one if count == 1 else text.many).format(n=count)
    if text.subject == "unit" and ids:
        names = list(dict.fromkeys(object_catalog.display_name(i) for i in ids))
        more = ", ..." if len(names) > WARNING_EXAMPLES else ""
        line += f" ({', '.join(names[:WARNING_EXAMPLES])}{more})"
    return count, f"  {line}."


def warning_summary(tables: game_dataset.DataModTables) -> str:
    """The status log's lines for a mod's derive warnings (GH #323): a header
    with simplified_count(), one line per kind by descending count, and where
    the technical text is for a bug report."""
    by_kind: dict[str, list] = {kind: [] for kind in WARNING_TEXT}
    for warning in tables.warnings:
        by_kind[_kind_of(warning)].append(warning)
    # sorted() is stable, so equal counts keep WARNING_TEXT's order.
    kind_lines = sorted((_kind_line(kind, found) for kind, found in by_kind.items() if found), key=lambda cl: -cl[0])
    return "\n".join([
        (
            f"Data mod {tables.title} loads simplified: DEscape shows {simplified_count(tables.warnings)} "
            "of its objects or terrains differently from the game."
        ),
        *(line for _count, line in kind_lines),
        (
            "Your scenario is saved as usual. For a bug report to the mod's author or DEscape, "
            f"the technical details are in {tables.tables_dir / dat_cache.META_NAME}."
        ),
    ])
