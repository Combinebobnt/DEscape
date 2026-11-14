"""ViewerWindow's files: game data, autosave, save, open, recent and dropped files, New Map, resize, load_scenario and close_scenario."""

from __future__ import annotations

import dataclasses
import gc
import time
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
)

from descape import (
    asset_source,
    autosave,
    debug_log,
    game_dataset,
    games,
    gc_hold,
    perf_trace,
    place_rotation,
    settings,
    terrain_units,
    ui_icons,
    unit_sprites,
)
from descape.autosave_dialog import RecoverAutosaveDialog
from descape.data_mod_menu import FROM_PATH
from descape.new_map_dialog import NewMapChoice, NewMapDialog
from descape.resize_dialog import ResizeDialog
from descape.resize_preview import preview_source
from descape.scenario_io import (
    BLANK_TEMPLATE_TILES,
    TEMPLATE_DIR,
    LoadedScenario,
    UnsupportedStructureVersion,
    is_in_game_profile,
    load_map_and_units,
    load_map_and_units_from_bytes,
    refresh_player_render_context,
    resolve_detecting_loops,
)
from descape.scenario_new import (
    LARGE_MAP_CONFIRM_TILES,
    NEW_MAP_UNIT_CONFIRM,
    STANDARD_MAP_SIZES,
    BlankFillError,
    BlankGenerationError,
    MapSizeError,
    blank_scenario_bytes,
)
from descape.scenario_resize import Anchor, ResizeRefusedError, resize_plan, resize_scenario
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.trigger_model import (
    TriggerEditsUnavailableError,
)
from descape.unit_model import (
    UnitEditsUnavailableError,
)
from descape.viewer_common import (
    HOVER_IDLE_TEXT,
)


def dropped_scenario_paths(urls: Sequence[QUrl]) -> list[Path]:
    """GH #221: the local .aoe2scenario files among dropped URLs, in drop order."""
    paths = [Path(url.toLocalFile()) for url in urls if url.isLocalFile()]
    return [path for path in paths if path.suffix.lower() == ".aoe2scenario"]


# Display-only sentinel assigned to LoadedScenario.path for a File > New map.
# Deliberately relative and non-existent: LoadedScenario.path is only ever read
# for display and for Save As's default name -- write_scenario() takes an
# explicit destination and never reads it (see that field's own comment in
# scenario_io.py).
UNTITLED_NAME = "Untitled.aoe2scenario"
UNTITLED_PATH = Path(UNTITLED_NAME)

# How long an autosave tick waits before looking again, when it arrived
# while the window was busy or mid-stroke. Short enough that a long paint
# session doesn't starve autosave for a whole interval.
AUTOSAVE_RETRY_MS = 5000


class _ViewerFileMixin:
    """ViewerWindow's file handling and game data (viewer.py's docstring has the split's rules)."""

    # -- game data (GH #67) ------------------------------------------------

    def _file_game_dataset(self) -> str:
        """The open file's own game: its header variant flag, the default game with no document."""
        return games.default().id if self.scenario is None else games.for_scenario(self.scenario).id

    def _shown_game_dataset(self) -> str:
        """Settings > Advanced > Game data's choice, or the file's own game under Auto."""
        choice = settings.get_game_data()
        return self._file_game_dataset() if choice == settings.GAME_DATA_DEFAULT else choice

    def game_data_lock_reason(self) -> str | None:
        """Why the open document may not be written, or None. Shown in the
        status bar while it applies. GH #67: a file is editable only with its
        own game's data shown (Return of Rome included), and never when its
        game is view-only (game_dataset.lock_reason())."""
        if self.scenario is None:
            return None
        return game_dataset.lock_reason(self._file_game_dataset(), self._shown_game_dataset())

    def _cross_game_refusal(self, block) -> str:
        """Why a clipboard block copied from the other game's file may not be pasted here, or "".
        Compares the two files' own games, not the shown data: a locked view's copy keeps its file's ids."""
        copied = getattr(block, "dataset", games.DEFAULT_ID)
        here = self._file_game_dataset()
        if copied == here:
            return ""
        return (
            f"Copied with {game_dataset.DISPLAY_NAMES[copied]} game data; this document is a "
            f"{game_dataset.DISPLAY_NAMES[here]} file, where the same ids are other objects."
        )

    def _with_file_game(self, block):
        """`block` stamped with the open file's own game, which its ids belong to (GH #67)."""
        return dataclasses.replace(block, dataset=self._file_game_dataset())

    def _refuse_locked_save(self) -> bool:
        reason = self.game_data_lock_reason()
        if reason is None:
            return False
        self._log_status(f"Save refused. {reason}", level=debug_log.WARNING)
        QMessageBox.information(self, "Read-only", reason)
        return True

    def _refuse_locked_edit(self) -> bool:
        """True (and says why) when an edit model is asked for on a locked document."""
        reason = self.game_data_lock_reason()
        if reason is not None:
            self._log_status(f"Edit refused. {reason}", level=debug_log.WARNING)
        return reason is not None

    def set_game_data(self, choice: str) -> None:
        """Settings > Advanced > Game data: saves `choice` (a settings.game_data_choices() id)
        and redraws the open document in it. Display only: the file's own flag is never
        changed, and a mismatch locks the document."""
        settings.set_game_data(choice)
        if game_dataset.set_active(self._shown_game_dataset()):
            self._refresh_game_assets()
        self._after_game_data_change()

    def _refresh_game_lists(self) -> None:
        """Every long-lived list built from game data: the Units catalog and its
        thumbnails, the terrain picker and its swatches, the wall and cliff menus, the AI list."""
        self.units_panel.reload_catalog()
        self.units_panel.refresh_thumbnails()
        # Fresh rows and cells: no refresh_swatches() pass needed after it.
        self.terrain_panel.reload_catalog()
        if self.mode == "terrain":
            self._apply_left_pane_floor()  # a mod's names can be wider
        self._refresh_unit_tool_menus()
        self._drop_ai_scan()

    def _refresh_game_assets(self) -> None:
        """After a dataset switch on an open document: the lists, the map, the
        unit pick index, the labels and the open mode's panel. The Find and
        Map Analysis dialogs close, since their rows name the other game's objects."""
        self._refresh_game_lists()
        self._close_analysis_dialog()
        self._close_find_dialog()
        if self.scenario is None:
            return  # Settings > Advanced with no document open
        # GH #263: the stored building art follows the data mod's civ list; refresh_map() draws it.
        refresh_player_render_context(self.scenario, self._pending_player_values())
        self.refresh_map()
        if self.map_view._unit_index is not None:
            self._rebuild_unit_index()
        self._refresh_player_labels()
        self.units_panel.refresh_portrait_civs()
        self.trigger_panel.refresh_portraits()
        self._update_info()

    def _after_game_data_change(self) -> None:
        """Re-gates everything the lock reaches: tools, edit actions, the open panel, the title and banner."""
        self._repopulate_players()
        self._repopulate_map_options()
        self._repopulate_diplomacy()
        self._repopulate_messages()
        self._update_tool_enabled()
        self._update_edit_actions()
        self._update_title()
        self._show_game_data_status()

    def _show_game_data_status(self) -> None:
        """The status bar's lock banner, and one line when the active game's own root (Return of Rome's art) is missing."""
        reason = self.game_data_lock_reason()
        self.game_data_banner.setText(reason or "")
        self.game_data_banner.setVisible(reason is not None)
        install = asset_source.get_install_path()
        shown = game_dataset.game()
        if install is not None and shown.root_subpath is not None and not shown.root(install).is_dir():
            name = shown.display_name
            self._log_status(f"{name} art is not installed: names and stats are {name}'s, units draw as coloured marks.")

    def _confirm_discard_changes(self) -> bool:
        """True if it's OK to proceed: no scenario loaded, no unsaved edits,
        or the user explicitly chose to discard them. Call before anything
        that would drop the current scenario -- open, close, or window close.
        A focused field's typed value commits first, so the prompt counts it."""
        self._commit_focused_field()
        if self.scenario is None or not self.edit_history.is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"{self.scenario.path.name} has unsaved changes. Discard them?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Discard

    def _save_as_start_path(self) -> str:
        if self._untitled:
            # Never the template's own directory (Path.home() is always
            # writable and never TEMPLATE_DIR).
            return str(Path.home() / UNTITLED_NAME)
        src = self.scenario.path
        # File > Open can reach the shipped template itself; without this,
        # the default would aim inside TEMPLATE_DIR, which write_scenario()
        # unconditionally refuses to write into.
        try:
            in_template = resolve_detecting_loops(src.parent) == TEMPLATE_DIR.resolve()
        except (OSError, RuntimeError):  # a symlink loop: only a start folder, so keep it as given
            in_template = False
        parent = Path.home() if in_template else src.parent
        return str(parent / src.name)

    def show_recover_autosave(self) -> None:
        """File > Recover from Autosave…. Modeless and reused, like the
        clipboard-history and analysis dialogs."""
        if self._recover_dialog is None:
            self._recover_dialog = RecoverAutosaveDialog(self, self._open_autosave_entry)
        else:
            self._recover_dialog.refresh()
        self._recover_dialog.show()
        self._recover_dialog.raise_()
        self._recover_dialog.activateWindow()

    def _open_autosave_entry(self, entry) -> None:
        """Open a slot as an UNTITLED document: save() then falls through to
        Save As, so a recovered document can never silently overwrite the
        original the user is trying to compare it against."""
        if not self._confirm_discard_changes():
            return
        if self._recover_dialog is not None:
            self._recover_dialog.close()
        self.load_scenario(entry.path, untitled=True, data_mod=entry.data_mod)
        if self.scenario is not None:
            self._log_status(f"Recovered {entry.path.name} as a new untitled document")

    def _edit_model_kwargs(self) -> dict:
        """The edit models write_scenario() needs, in one place. save(),
        save_as() and the autosave tick are three call sites that have to
        stay in lockstep forever, and the last one already cost a real gap
        once (an autosave that omitted messages= would have silently dropped
        every Messages-mode edit). Adding a model here reaches all three."""
        return {
            "triggers": self.trigger_edits,
            "options": self.option_edits,
            "units": self.unit_edits,
            "messages": self.message_edits,
        }

    def _autosave_doc_key(self) -> str:
        """This document's rotation key -- its resolved path, or its own
        doc_id while untitled."""
        path = None if (self._untitled or self.scenario is None) else self.scenario.path
        return autosave.doc_key(path, self._doc_id)

    def _stroke_in_progress(self) -> bool:
        """True between begin_stroke() and commit_stroke(), or while
        MapView is mid-drag. Serializing here would write a half-applied
        stroke."""
        return self.edit_history.in_stroke or self.map_view._stroke_active

    def _reset_autosave_timer(self) -> None:
        """Re-arm from the current settings -- called at startup and by the
        Saving tab's enabled/interval handlers. Retention and location need
        no restart; they are read fresh on each tick."""
        self._autosave_timer.stop()
        if settings.get_autosave_enabled():
            self._autosave_timer.start(settings.get_autosave_interval_min() * 60 * 1000)

    def _autosave_tick(self) -> None:
        """Write this document's next recovery slot, if it is worth writing
        and this is a safe moment to write it.

        The three gates are all free, and each earns its place. is_dirty:
        write_scenario()'s own no-op short circuit sits AFTER full
        serialization, so letting a clean document reach it would burn the
        whole ~0.9s (measured, largest corpus file) to discover there was
        nothing to do. The content_key comparison is the same idea one level
        in: a document edited once and then left alone stays dirty forever.
        It ignores view-only records, so a ruler-only cursor move is no change.
        _busy/mid-stroke is a correctness gate rather than an optimization
        -- a QTimer callback can be delivered inside any of viewer.py's
        processEvents() calls, i.e. mid-load or mid-render.

        **The settle rule.** The retry timer connects back here, so during a
        long paint session it re-arms every 5s and would otherwise fire on
        the first clear tick, which can land exactly as the user begins the
        next stroke. A retry that saw a stroke since it was armed re-arms
        once more instead of writing: one extra retry wait after the last
        stroke, and no write ever begins inside a paint burst.

        Never calls mark_saved() and never touches saved_at_cursor. A
        recovery slot does not mean the document matches its real file;
        clearing either would drop the title's "*" AND make
        _confirm_discard_changes() return True, so the user would close
        without a prompt and lose the work. It calls write_scenario()
        directly rather than save() for the same class of reason: save()
        routes an untitled document into a modal Save As dialog.

        Never commits a focused field's typed value, unlike save(): a timer
        that did would split one typed edit into undo steps the user never
        chose and could rebuild the trigger form under the cursor. A slot
        goes without the half-typed value.
        """
        if self.scenario is None or not settings.get_autosave_enabled():
            return
        if self.game_data_lock_reason() is not None:  # GH #67: a locked document is never written
            return
        if not self.edit_history.is_dirty:
            return
        if self.edit_history.content_key == self._autosaved_content:
            return
        if self._busy or self._stroke_in_progress():
            self._stroke_seen_since_retry = True
            self._autosave_retry_timer.start(AUTOSAVE_RETRY_MS)
            return
        if self._stroke_seen_since_retry:
            self._stroke_seen_since_retry = False
            self._autosave_retry_timer.start(AUTOSAVE_RETRY_MS)
            return
        with perf_trace.phase("autosave"):
            self._autosave_write()

    def _autosave_write(self) -> None:
        """_autosave_tick's write, once every gate has passed."""
        try:
            # Inside the try so a path error here is logged, never raised out of a timer slot.
            key = self._autosave_doc_key()
            source = None if self._untitled else self.scenario.path
            custom = settings.get_autosave_dir()
            central, reason = autosave.central_dir_with_refusal(custom)
            slot = autosave.slot_path(
                key, self.scenario.path.name, source, settings.get_autosave_location(), central=central
            )
            fallback = (custom, reason) if reason is not None else None
            if not custom:  # a saved folder refused at startup reads as "", so ask settings why
                fallback = settings.get_autosave_dir_refusal()
            if fallback is None or slot.parent != central:
                self._autosave_fallback_logged = None
            elif fallback != self._autosave_fallback_logged:
                self._autosave_fallback_logged = fallback
                self._log_status(
                    f"Autosave folder {fallback[0]} can't hold autosaves: {fallback[1]}; wrote to the default folder",
                    level=debug_log.WARNING,
                )
            # Never for a custom folder: a vanished one falls back above rather than reappearing.
            if slot.parent != central or central == autosave.autosave_dir():
                slot.parent.mkdir(parents=True, exist_ok=True)
            # backup=False is mandatory, not a preference: the default would
            # put a .bak/.orig pair beside every rotating slot.
            write_scenario(self.scenario, slot, backup=False, **self._edit_model_kwargs())
        except Exception as e:  # noqa: BLE001 -- GUI boundary, reports below
            # Swallowed to a log line, never a modal. An autosave that pops a
            # QMessageBox mid-edit is worse than one that quietly fails; the
            # user's real file is untouched either way.
            self._log_status(f"Autosave failed: {type(e).__name__}: {e}", level=debug_log.ERROR)
            return
        autosave.record(
            key, slot, str(self.scenario.path), untitled=self._untitled, data_mod=self.data_mod_menu.identity()
        )
        autosave.rotate(key, settings.get_autosave_retention())
        self._autosaved_content = self.edit_history.content_key
        self._log_status(f"Autosaved to {slot.name}")

    def save(self) -> None:
        if self.scenario is None or self._refuse_locked_save():
            return
        self._commit_focused_field()
        if self._untitled:
            # No path to write back to yet -- same first-save-needs-a-name
            # case Save As already handles.
            self.save_as()
            return
        path = str(self.scenario.path)
        autosave_key = self._autosave_doc_key()
        try:
            result = write_scenario(
                self.scenario,
                path,
                backup=settings.get_backups_enabled() or is_in_game_profile(path),
                **self._edit_model_kwargs(),
            )
        except WriteBlockedError as e:
            self._log_status(f"Save blocked: {e}", level=debug_log.WARNING)
            QMessageBox.critical(self, "Save blocked", str(e))
            return
        except Exception as e:  # noqa: BLE001 -- GUI boundary, reports below
            self._log_status(f"Save failed: {type(e).__name__}: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.edit_history.mark_saved()
        self._update_title()
        # The real file now holds this state, so its recovery slots are
        # obsolete -- leaving them would let Recover offer something OLDER
        # than what is on disk.
        autosave.discard(autosave_key)
        self._autosaved_content = self.edit_history.content_key
        if not result.wrote:
            self._log_status("No changes to save")
        else:
            names = ", ".join(p.name for p in result.backups)
            suffix = f" (backed up {names})" if names else ""
            self._log_status(f"Saved to {path}{suffix}")

    def save_as(self) -> None:
        if self.scenario is None or self._refuse_locked_save():
            return
        self._commit_focused_field()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save scenario as", self._save_as_start_path(), "AoE2 Scenario (*.aoe2scenario)"
        )
        if not path:
            return
        # Captured BEFORE the retarget below, which reassigns scenario.path and
        # clears _untitled: from that point on this document keys by its new
        # path, so discarding a freshly-computed key would drop the wrong
        # (empty) one and leak the untitled slots into Recover under a stale
        # "Untitled" label.
        autosave_key = self._autosave_doc_key()
        try:
            result = write_scenario(
                self.scenario,
                path,
                backup=settings.get_backups_enabled() or is_in_game_profile(path),
                **self._edit_model_kwargs(),
            )
        except WriteBlockedError as e:
            self._log_status(f"Save blocked: {e}", level=debug_log.WARNING)
            QMessageBox.critical(self, "Save blocked", str(e))
            return
        except Exception as e:  # noqa: BLE001 -- GUI boundary, reports below
            self._log_status(f"Save failed: {type(e).__name__}: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        # Save As always retargets the open document to the file just written,
        # matching the usual editor convention (Notepad, VS Code, etc.): the
        # window is now editing that file, not the one it was opened from.
        self.scenario.path = Path(path)
        self._untitled = False
        # GH #58: the new path opens with the data mod it was saved under.
        self.data_mod_menu.remember_current()
        # GH #215: a new map or a renamed copy is otherwise missing from Open Recent until reopened.
        settings.add_recent_file(self.scenario.path)
        self._rebuild_recent_files_menu()
        self._update_info()  # the info panel's "File:" line just changed
        self.edit_history.mark_saved()
        self._update_title()
        # The real file now holds this state, so its recovery slots are
        # obsolete -- leaving them would let Recover offer something OLDER
        # than what is on disk.
        autosave.discard(autosave_key)
        self._autosaved_content = self.edit_history.content_key
        if not result.wrote:
            self._log_status("No changes to save")
        else:
            names = ", ".join(p.name for p in result.backups)
            suffix = f" (backed up {names})" if names else ""
            self._log_status(f"Saved to {path}{suffix}")

    def open_file(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open scenario", "", "AoE2 Scenario (*.aoe2scenario)"
        )
        if not path:
            return
        self.load_scenario(Path(path))

    def _rebuild_recent_files_menu(self) -> None:
        """Repopulates File > Open Recent from settings.get_recent_files().
        Called at menu-bar build time and again after every successful open
        -- QMenu keeps no live binding to the underlying list, so this is
        the one place the widget is kept in sync with it. Filters out
        entries whose file no longer exists rather than pruning them from
        the persisted list, so a file on a removable drive reappears once
        it's back."""
        self.recent_menu.clear()
        self.clear_recent_action = None
        paths = [p for p in settings.get_recent_files() if Path(p).is_file()]
        if not paths:
            empty_action = QAction("(No recent files)", self)
            empty_action.setToolTip("Scenarios you open are listed here")
            empty_action.setEnabled(False)
            self.recent_menu.addAction(empty_action)
            return
        for path_str in paths:
            action = QAction(Path(path_str).name, self)
            action.setToolTip(path_str)
            # Marks this as a real file entry, distinct from the "Clear
            # Recent Files" action below and the disabled placeholder above
            # -- both of which would otherwise be indistinguishable from a
            # file action by toolTip() alone, since Qt falls back to the
            # (mnemonic-stripped) text when no tooltip was set explicitly.
            action.setData(path_str)
            action.triggered.connect(lambda checked=False, p=path_str: self._open_recent(p))
            self.recent_menu.addAction(action)
        self.recent_menu.addSeparator()
        clear_action = QAction("&Clear Recent Files", self)
        clear_action.setToolTip("Empty this list. The files themselves are not touched")
        clear_action.setIcon(ui_icons.icon(ui_icons.CLEAR_RECENT_ICON))
        clear_action.triggered.connect(self._clear_recent_files)
        self.recent_menu.addAction(clear_action)
        self.clear_recent_action = clear_action

    def _open_recent(self, path_str: str) -> None:
        if not self._confirm_discard_changes():
            return
        self.load_scenario(Path(path_str))

    def _drop_paths(self, event) -> list[Path]:
        """The scenario files a drag carries, or [] when it can't open one now."""
        if self._busy or not event.mimeData().hasUrls() or not event.possibleActions() & Qt.CopyAction:
            return []
        return dropped_scenario_paths(event.mimeData().urls())

    @staticmethod
    def _accept_as_copy(event) -> None:
        # Never Move (Shift-drag): Explorer deletes the source after a Move result, and opening consumes nothing.
        event.setDropAction(Qt.CopyAction)
        event.accept()

    def _open_dropped(self, paths: list[Path]) -> None:
        if self._busy or not self._confirm_discard_changes():
            return
        before = self.scenario
        self.load_scenario(paths[0])
        if len(paths) > 1 and self.scenario is not before:
            message = f"Opened {paths[0].name}; skipped {len(paths) - 1} other dropped file(s)"
            self._log_status(message)
            self.statusBar().showMessage(message)

    def _clear_recent_files(self) -> None:
        settings.clear_recent_files()
        self._rebuild_recent_files_menu()

    def new_map(self, tiles: int = BLANK_TEMPLATE_TILES) -> None:
        """Starts a blank `tiles`x`tiles` map by generating it in memory from
        the shipped 120x120 donor template (descape.scenario_new.
        blank_scenario_bytes -- the file_new_default keybind's 120x120, or
        any size in [MIN_MAP_TILES, MAX_MAP_TILES] via new_map_dialog(),
        which calls _create_new_map() itself) and loading the result through the normal load path,
        then marking the document untitled -- see load_scenario()'s `untitled`
        handling. Not built from AoE2ScenarioParser's own from_default():
        that would need a second write path through the library's own
        (unverified, non-byte-stable) serializer instead of the already
        in-game-verified byte-patch path this reuses unchanged.

        The new map inherits the donor's FileHeader verbatim (creator name,
        timestamps) and DataHeader.filename -- the byte-patch splice can't
        rewrite those by design: filename is DataHeader's terminal,
        length-prefixed field, and rewriting it to a different-length name
        would shift every section offset after it. Inherent to reusing a
        donor rather than a defect in it.
        """
        if not self._confirm_discard_changes():
            return
        self._create_new_map(tiles)

    def new_map_dialog(self) -> None:
        """File > New Map… (Ctrl+N, GH #243). Confirms discard once, opens
        NewMapDialog at the last choices (saved again on OK), then asks at
        most once before generating. A size above LARGE_MAP_CONFIRM_TILES that
        is not one of STANDARD_MAP_SIZES asks; a preset size, 480 included,
        even typed into Custom, never does for its size alone: a preset is a
        size already proven in-game, so a confirm there would just be a nag,
        while any other size has no such history. Trees or Eye candy expected
        to plant more than NEW_MAP_UNIT_CONFIRM units (a Ludicrous forest)
        asks too. With nothing to plant, the size question is a plain
        Yes/Cancel; with planting it is _ask_new_map_confirm(), whose third
        answer creates the map bare.

        Calls _create_new_map() directly rather than new_map(), which would
        ask the discard question a second time."""
        if self._new_map_panel is not None and self._new_map_panel.isVisible():
            return  # already open
        if not self._confirm_discard_changes():
            return
        initial = NewMapChoice.from_dict(settings.get_new_map_choice())
        dialog = self._new_map_panel
        if dialog is None or dialog.built_for != NewMapDialog.build_key():
            if dialog is not None:
                dialog.deleteLater()
            dialog = self._new_map_panel = NewMapDialog(self, initial)
        else:
            dialog.set_initial(initial)
        accepted = dialog.exec_in(self) == QDialog.Accepted
        choice = dialog.choice
        if not accepted or choice is None:
            return
        settings.set_new_map_choice(choice.as_dict())
        trees, doodads, planted = self._new_map_planting(choice)
        large = choice.tiles > LARGE_MAP_CONFIRM_TILES and choice.tiles not in STANDARD_MAP_SIZES
        large_text = (
            f"{choice.tiles}×{choice.tiles} is larger than the {LARGE_MAP_CONFIRM_TILES}×"
            f"{LARGE_MAP_CONFIRM_TILES} standard size. Large maps take "
            "longer to create and render, and use significantly more "
            "memory."
        )
        if not (trees or doodads):
            if large:
                reply = QMessageBox.question(
                    self,
                    "Large map",
                    f"{large_text} Create it anyway?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if reply != QMessageBox.Yes:
                    return
        elif large or planted > NEW_MAP_UNIT_CONFIRM:
            reasons = [large_text] if large else []
            if planted > NEW_MAP_UNIT_CONFIRM:
                kind = "trees and eye candy" if trees and doodads else "trees" if trees else "eye candy objects"
                reasons.append(
                    f"It gets about {planted:,} {kind}. Creating it may take a while, and at "
                    "Ludicrous size it uses about 1 GB more memory than a bare map."
                )
            bare_kind = "trees or eye candy" if trees and doodads else "trees" if trees else "eye candy"
            answer = self._ask_new_map_confirm("\n\n".join([*reasons, "Create it anyway?"]), bare_label=f"Create without {bare_kind}")
            if answer == "cancel":
                return
            if answer == "bare":
                trees = doodads = False  # this map only: the saved choice keeps its boxes
        self._create_new_map(
            choice.tiles,
            terrain_id=choice.terrain_id,
            elevation=choice.elevation,
            game_data=choice.game_data,
            trees=trees,
            doodads=doodads,
        )

    def _new_map_planting(self, choice: NewMapChoice) -> tuple[bool, bool, int]:
        """(trees, eye candy, expected unit count) for a New Map… choice: only
        the ticked kinds its terrain can place, read under its game's tables."""
        if not (choice.trees or choice.eye_candy):
            return False, False, 0
        with game_dataset.activated(choice.game_data):
            trees = choice.trees and terrain_units.is_tree_terrain(choice.terrain_id)
            doodads = choice.eye_candy and terrain_units.is_doodad_terrain(choice.terrain_id)
            planted = terrain_units.expected_unit_count(
                choice.terrain_id, choice.tiles * choice.tiles, trees=trees, doodads=doodads
            )
        return trees, doodads, planted

    def _ask_new_map_confirm(self, text: str, *, bare_label: str) -> str:
        """New Map…'s one question when it plants: "create", "bare" (create
        it planting nothing, the `bare_label` button) or "cancel"."""
        box = QMessageBox(QMessageBox.Question, "New Map", text, QMessageBox.NoButton, self)
        create = box.addButton("Create", QMessageBox.AcceptRole)
        bare = box.addButton(bare_label, QMessageBox.ActionRole)
        cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec_()
        clicked = box.clickedButton()
        box.deleteLater()
        if clicked is create:
            return "create"
        if clicked is bare:
            return "bare"
        return "cancel"

    def _create_new_map(
        self,
        tiles: int,
        *,
        terrain_id: int = 0,
        elevation: int = 0,
        game_data: str = game_dataset.AOE2,
        trees: bool = False,
        doodads: bool = False,
    ) -> None:
        """The generate-and-load half of File > New Map…, shared by new_map()
        and new_map_dialog() -- both of which have already run
        _confirm_discard_changes() themselves before calling this, so it
        does no confirming of its own. game_data picks the donor, and the
        load sets the active dataset from the result's own header.

        trees / doodads plant into the generated file (scenario_new), which
        can take seconds and switches to game_data's tables while it runs:
        the window is disabled under a wait cursor with a status line,
        painted before the switch, and restored before load_scenario(), which
        shows its own. Never sets self._busy, which would make load_scenario()
        return at once."""
        try:
            t_generate0 = time.perf_counter()
            if trees or doodads:
                self.statusBar().showMessage("Planting trees…" if trees else "Planting eye candy…")
                self.setEnabled(False)
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    QApplication.processEvents()
                    # No warm may resume across the switch: it refills tables they read.
                    if game_data != game_dataset.active():
                        self._cancel_warms()
                    data = blank_scenario_bytes(
                        tiles,
                        terrain_id=terrain_id,
                        elevation=elevation,
                        game_data=game_data,
                        trees=trees,
                        doodads=doodads,
                    )
                finally:
                    QApplication.restoreOverrideCursor()
                    self.setEnabled(True)
            else:
                data = blank_scenario_bytes(tiles, terrain_id=terrain_id, elevation=elevation, game_data=game_data)
            generate_elapsed = time.perf_counter() - t_generate0
        except (MapSizeError, BlankFillError, BlankGenerationError) as e:
            self.statusBar().clearMessage()
            self._log_status(f"New Map failed: {type(e).__name__}: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "New Map failed", str(e))
            return
        self.load_scenario(UNTITLED_PATH, untitled=True, data=data, generate_elapsed=generate_elapsed)

    def _confirm_resize_unsaved(self) -> bool:
        """True to go on with a resize. Unlike _confirm_discard_changes(),
        nothing is discarded: the resize carries every unsaved edit into the
        resized map (scenario_write.build_patched_body()), but the undo
        history goes, so a dirty document still asks once. A focused field's
        typed value commits first, so the prompt counts it."""
        self._commit_focused_field()
        if not self.edit_history.is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"{self.scenario.path.name} has unsaved changes. Resizing keeps them in the "
            "resized map, still unsaved, but clears the undo history. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Yes

    def _resize_plan_for(self, width: int, height: int, anchor: Anchor):
        return resize_plan(self.scenario, width, height, anchor)

    def resize_map(self) -> None:
        """Map > Resize Map…: the unsaved-changes prompt (once), the size and
        anchor dialog, a confirm for anything destructive (tiles cut off or
        objects deleted, since a resize cannot be undone), then
        _apply_resize()."""
        if self.scenario is None or self._busy:
            return
        if not self._confirm_resize_unsaved():
            return
        mm = self.scenario.map_manager
        isometric = self._terrain_style != "flat" or self.iso_action.isChecked()
        dialog = ResizeDialog(
            self,
            mm.map_width,
            mm.map_height,
            self._resize_plan_for,
            last=self._resize_defaults,
            preview=preview_source(self.scenario, isometric=isometric),
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self._resize_defaults = dialog.state()  # the dialog's own OK keeps it, even if the confirm is declined
        (width, height), anchor, plan = dialog.new_size, dialog.anchor, dialog.plan
        if plan is None or plan.refusal is not None:
            return
        if plan.destructive:
            reply = QMessageBox.question(
                self,
                "Resize map",
                f"{plan.summary()}\n\nResize anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
        self._apply_resize(width, height, anchor)

    def _apply_resize(self, width: int, height: int, anchor: Anchor) -> None:
        """Builds the resized document from the current one, pending edits
        included, then replaces the document through load_scenario(), which
        resets the history (every flat tile index in it is stale) and drops
        the edit models. The resize's own unit/trigger models are attached
        after it returns, and the document stays dirty until the next save."""
        old = self.scenario
        try:
            result = resize_scenario(
                old,
                width,
                height,
                anchor,
                triggers=self.trigger_edits,
                options=self.option_edits,
                units=self.unit_edits,
                messages=self.message_edits,
            )
        except (ResizeRefusedError, WriteBlockedError, UnitEditsUnavailableError, TriggerEditsUnavailableError) as e:
            self._log_status(f"Resize failed: {type(e).__name__}: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Resize failed", str(e))
            return
        self.load_scenario(old.path, untitled=self._untitled, data=result.data, loaded=result.loaded)
        if self.scenario is not result.loaded:
            return  # the load itself failed and already reported why
        self.unit_edits = result.unit_edits
        self.trigger_edits = result.trigger_edits
        self.edit_history.saved_at_cursor = None
        self._update_edit_actions()
        self._update_title()
        plan = result.plan
        self._log_status(
            f"Resized {plan.old_w}x{plan.old_h} to {width}x{height} anchored {anchor.label}: "
            f"{result.units_deleted} objects deleted, triggers {plan.triggers}"
        )

    def _freeze_loaded_document(self) -> None:
        """One full collection, then gc.freeze(): the loaded document's ~500k
        tracked objects move to the permanent generation, so no later full
        collection walks them (21-23 ms each on old-allies, 2026-09-28, at any moment of
        a session). Undone by gc.unfreeze() at the next load, close_scenario()
        and closeEvent(), so a closed document is still freed.

        Runs after the load's _start_level_warm(), so it replaces that warm's
        gc hold (gc_hold) with this collection, then holds again for the warm."""
        gc_hold.reset()
        t0 = time.perf_counter()
        gc.collect()
        gc.freeze()
        self._load_gc_collect_s = time.perf_counter() - t0
        debug_log.log(f"Load gc.collect before freeze: {self._load_gc_collect_s * 1000:.1f} ms")
        if any(w.is_active for w in (self._level_warmer, self._margin_warmer, self._load_warmer)):
            gc_hold.engage()

    def load_scenario(
        self,
        path: Path,
        *,
        untitled: bool = False,
        data: bytes | None = None,
        generate_elapsed: float | None = None,
        loaded: LoadedScenario | None = None,
        data_mod=FROM_PATH,
    ) -> None:
        # generate_elapsed: File > New Map's blank_scenario_bytes() time, spent
        # before this call; reported as `generate` and counted in the totals.
        # loaded: `data` already parsed (a map resize, which remaps units and
        # triggers on it before it is shown), adopted instead of parsed again.
        # data_mod: a recovered autosave's data mod (GH #58), else FROM_PATH.
        # Loading + rendering a large map is a multi-second blocking call
        # (see _render_current()'s own docstring) -- the log line and status
        # bar message below are queued but not actually painted until
        # control returns to the event loop, so without this
        # setOverrideCursor()+processEvents() pair the window would just sit
        # frozen with no visible indication anything is happening until the
        # whole load completes. self._busy (see its own comment in
        # __init__) plus setEnabled(False) together guard against the same
        # reentrancy risk _render_current() documents for its own
        # processEvents() call -- e.g. a second Open dispatched from inside
        # this one's processEvents() while self.scenario is still being
        # reassigned. A true no-op, not just queued: checked before
        # anything else runs.
        if self._busy:
            return
        self._busy = True
        # Before the processEvents() below, where the previous document's build could still fire.
        self._unit_model_prebuild.cancel()
        self._close_analysis_dialog()
        self._close_find_dialog()
        # Dropped before anything else: a previous load whose paint never
        # came (or came late) must not have its line attributed to this file.
        self._paint_report_timer.stop()
        self._pending_paint_report = None
        try:
            self._log_status(f"Loading scenario: {path}")
            self.statusBar().showMessage(
                "Creating new map..." if untitled else f"Loading {path.name}..."
            )
            self.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            # The previous document's frozen objects become collectable again once this load drops them.
            gc.unfreeze()
            gc_hold.reset()
            try:
                t_parse0 = time.perf_counter()
                if loaded is not None:
                    self.scenario = loaded
                elif data is not None:
                    self.scenario = load_map_and_units_from_bytes(data, path)
                else:
                    self.scenario = load_map_and_units(path)
                parse_elapsed = time.perf_counter() - t_parse0
            except UnsupportedStructureVersion as e:
                # Named cause instead of the library's raw exception text: a
                # v1.32/v1.35 file is unopenable by design, not a broken file.
                self.statusBar().clearMessage()
                self._log_status(
                    f"Failed to load {path}: unsupported scenario version {e.scenario_version}", level=debug_log.ERROR
                )
                QMessageBox.critical(self, "Failed to load", str(e))
                return
            except Exception as e:  # noqa: BLE001 -- GUI boundary, reports below
                self.statusBar().clearMessage()
                self._log_status(f"Failed to load {path}: {type(e).__name__}: {e}", level=debug_log.ERROR)
                QMessageBox.critical(self, "Failed to load", f"{type(e).__name__}: {e}")
                return
            finally:
                QApplication.restoreOverrideCursor()
                self.setEnabled(True)

            # Only reached on a successful load, so a failed File > New can never
            # mislabel whatever was previously open (self.scenario still holds it,
            # untouched, if the try block above returned early).
            self._untitled = untitled
            # GH #67: before the first paint, the game data Settings > Advanced picks for this file.
            switched = game_dataset.set_active(self._shown_game_dataset())
            # GH #58: and the data mod remembered for it; untitled keeps the session's.
            if self.data_mod_menu.for_document(path, untitled=untitled, remembered=data_mod):
                switched = True
            if switched:
                self._refresh_game_lists()
                # GH #263: the file parsed under the previous data mod's civ list. No pending
                # edits: self.option_edits still belongs to the previous document here.
                refresh_player_render_context(self.scenario, {})
            # TASK-438: a mod enabled or disabled in the game since the last scan; its redraw is this load's.
            self._pick_up_mod_changes(loading=True)
            # After the data switch and the pick-up, whose generation bumps would discard it (TASK-031.47).
            self._start_sprite_prefetch()
            if untitled:
                self.scenario.path = UNTITLED_PATH
            else:
                settings.add_recent_file(path)
                self._rebuild_recent_files_menu()
            display_name = self.scenario.path.name

            self._drop_erase_stroke()
            self._drop_unit_brush_stroke()
            self.edit_history.reset()
            # A new document is a new autosave key and a fresh "never
            # autosaved" state; the untitled key rides _doc_id, so reusing
            # the old one would rotate a different document's slots.
            self._doc_id = uuid4().hex
            self._autosaved_content = None
            self._autosave_fallback_logged = None
            self._stroke_seen_since_retry = False
            # A lift's base is the old history's, which an empty new one could match.
            self._set_lift(None)
            # Dropped alongside the history it pushes records onto: a model
            # held across a document switch would splice the previous file's
            # trigger bytes into this one, patch the previous file's option
            # offsets into it, or splice the previous file's unit bytes in.
            self.trigger_edits = None
            # _trigger_gen is not per document, so a kept cache would serve the previous file's effects.
            self._trigger_range_cache = None
            self.map_view.set_trigger_range_effects(None)
            self.option_edits = None
            self.unit_edits = None
            self.message_edits = None
            # Map-relative, like the hover position -- a region rect from
            # whatever was open before must not outlive it (the new map may
            # not even be big enough to contain it). The clipboard is NOT
            # reset here; see close_scenario()'s own comment on why it
            # survives.
            self._region = None
            # The keys outlive a mode switch, never a document (TASK-359). Before the disarm and
            # the render, which read _needs_unit_index(); set_source() drops MapView's copy.
            self._selection = []
            self.units_panel.show_selection_count(0)
            # The trigger it named belongs to the old document, as do the picker and the references.
            self._trigger_overlay_ref = None
            self.disarm_unit_picker()
            self._unit_ref_index = None
            # GH #172: a pending facing or variant is this document's, never the next one's.
            self._place_rotation = place_rotation.PlaceRotation()
            self._place_ghost_memo = None
            # TASK-327: the group grid holds the old document's units.
            self._reset_drag_rule_memos()

            # Opened before the render: the first paint can fire inside
            # _render_current()'s processEvents(), and counts as this load's.
            self._pending_paint_report = {
                "generate": generate_elapsed or 0.0, "parse": parse_elapsed, "prepare": 0.0, "paint": 0.0,
                "paints": 0, "max_paint": -1.0, "mip": 0, "ready": False,
            }
            # Renders at whichever Terrain Style was already selected --
            # File > Open doesn't reset it back to Flat. _render_current()
            # pushes its own wait cursor/setEnabled(False) for this step,
            # nested safely inside this method's own (see both docstrings).
            # The "Loading..." status bar message deliberately stays up
            # through this render step too, not just the parse above --
            # cleared only once everything is actually done, right below.
            elapsed, tile_px = self._render_current()
            self._pending_paint_report["prepare"] = elapsed
            # Here rather than inside _render_current(): this is the one
            # render path that opens a document the user is about to zoom
            # around in. _render_current()'s other callers are re-renders
            # that cancel a warm instead (see its own first line), re-armed
            # by the next viewport fire, as this load's warm is restarted by it.
            self._start_level_warm()
            self._freeze_loaded_document()
            self._schedule_unit_model_prebuild()
            mm = self.scenario.map_manager
            # Before _update_info(), whose stats combo rebuild reads these labels.
            self._refresh_player_labels()
            self.units_panel.refresh_portrait_civs()
            self._update_info()
            # Only when the panel is actually on screen -- otherwise opening a
            # map would parse a Triggers section nobody asked to see.
            if self.mode == "triggers":
                self._show_triggers()
            else:
                self.trigger_panel.clear_document()
            # Its own if/else rather than an elif chained onto the one above:
            # chaining would leave whichever panel lost the race populated
            # with the previous document.
            if self.mode == "map_options":
                self._show_map_options()
            else:
                self.map_options_panel.clear_document()
            # Its own if/else too, for the same reason as the map_options
            # pair above -- test_opening_a_second_map_while_in_the_mode_
            # repopulates pins this for Players mode as well.
            if self.mode == "players":
                self._show_players()
            else:
                self.players_panel.clear_document()
            # Its own if/else too, for the same reason as the two pairs
            # above.
            if self.mode == "diplomacy":
                self._show_diplomacy()
            else:
                self.diplomacy_panel.clear_document()
            # Its own if/else too, for the same reason as the pairs above.
            if self.mode == "messages":
                self._repopulate_messages()
            else:
                self.messages_panel.clear_document()
            self._update_tool_enabled()
            self._update_edit_actions()
            self._update_title()
            self.statusBar().clearMessage()
            self._show_game_data_status()
            total_elapsed = (generate_elapsed or 0.0) + parse_elapsed + elapsed
            generate_part = "" if generate_elapsed is None else f"generate {generate_elapsed:.2f}s, "
            self._log_status(
                f"{'Created' if untitled else 'Loaded'} {display_name} "
                f"({mm.map_width}x{mm.map_height} tiles, "
                f"{sum(len(u) for u in self.scenario.unit_manager.units):,} units, "
                f"tile_px={tile_px}, style={self._style_log_label}) "
                f"in {total_elapsed:.2f}s ({generate_part}parse {parse_elapsed:.2f}s, prepare {elapsed:.2f}s)"
            )
            if mod_line := self._mod_art_status():
                self._log_status(mod_line)
            if (grade := self.map_view.colour_grade) is not None:
                self._log_status(f"Colour mood: {grade.name}, previewed (View > Layers > Colour Mood)")
            # Starts the background walk of every active mod file once per mod-art change (GH #272).
            unit_sprites.sprite_reach()
            # Only now, with that line printed, may the follow-up drain. No
            # paint at all (a window never shown) means no second line.
            self._pending_paint_report["ready"] = True
            if self._pending_paint_report["paints"]:
                self._paint_report_timer.start()
            if not self.scenario.terrain_write_supported:
                self._log_status(
                    f"Warning: {display_name}'s terrain block failed load-time verification -- "
                    "Terrain mode's terrain/elevation tools and Save/Save As are disabled for "
                    "this file (still fully viewable).",
                    level=debug_log.WARNING,
                )
            elif not self.scenario.map_is_square:
                # Non-square is NOT actually supported today, despite what this message
                # used to claim: AoE2ScenarioParser's MapManager.map_size raises
                # ValueError("Map is not a square...") from tile.x/tile.y/get_tile(),
                # reached at ~15 call sites across render.py/viewer.py/elevation_tools.py/
                # batch_api.py -- so a non-square map fails during the render this same
                # load_scenario() call triggers below, before this note would ever help.
                # A possible follow-up: deriving tile coordinates from the terrain
                # index instead of these library properties.
                self._log_status(
                    f"Note: {display_name} is a non-square map ({mm.map_width}x{mm.map_height}) -- "
                    "non-square maps are not supported and will likely fail to render.",
                    level=debug_log.WARNING,
                )
        finally:
            self._busy = False

    def close_scenario(self) -> None:
        if self.scenario is None:
            return
        if not self._confirm_discard_changes():
            return
        name = self.scenario.path.name
        self.scenario = None
        self._close_analysis_dialog()
        self._close_find_dialog()
        self._untitled = False
        # Dropped alongside the cache it was warming -- otherwise a closed
        # document keeps ticking, holding the whole scenario alive.
        self._cancel_warms()
        self._stop_placeholder_timers()
        self._unit_model_prebuild.cancel()
        self._cache = None
        self._iso_elevations, self._iso_proj = None, None
        # Frozen at load; without this the closed document is never collected.
        gc.unfreeze()
        gc_hold.reset()
        self._drop_erase_stroke()
        self._drop_unit_brush_stroke()
        self.edit_history.reset()
        # The trigger and entry clipboards stay (GH #3's cross-document paste), and with them the
        # live Trigger/ce objects external_targets and a cut block's inbound pin. Accepted (GH #137).
        self._doc_id = uuid4().hex
        self._autosaved_content = None
        self._autosave_fallback_logged = None
        self._stroke_seen_since_retry = False
        self._autosave_retry_timer.stop()
        self._set_lift(None)
        self.trigger_edits = None
        self._trigger_range_cache = None
        self.map_view.set_trigger_range_effects(None)
        self.option_edits = None
        self.unit_edits = None
        self.message_edits = None
        self._trigger_overlay_ref = None
        # TASK-327: the group grid holds the closed document's units.
        self._reset_drag_rule_memos()
        # The keys outlive a mode switch, never a document (TASK-359); clear_image() drops MapView's copy.
        self._selection = []
        self.units_panel.show_selection_count(0)
        self.disarm_unit_picker()
        self._unit_ref_index = None
        self.map_view.clear_image()
        self._refresh_colour_mood_preview()
        # No edit tool has anything to act on with no map open; forcing Pan
        # (rather than just disabling the edit tools) keeps MapView's own
        # tool state in sync too -- same mechanism on_mode_changed already
        # uses when Terrain mode becomes unavailable.
        self.pan_action.setChecked(True)
        for title in ("File", "Terrain (top 8)", "Technical"):
            self._set_info_children(title, [])
        self._repopulate_stats_players()  # scenario is None: clears and disables the combo
        self._refresh_player_labels()  # back to the bare defaults, with no swatches
        self.units_panel.refresh_portrait_civs()  # GAIA's portraits, with no document
        self.trigger_panel.clear_document()
        self.map_options_panel.clear_document()
        self.players_panel.clear_document()
        self.diplomacy_panel.clear_document()
        self.messages_panel.clear_document()
        self._push_instruction_overlay()
        self._show_game_data_status()
        self._set_hover_text(HOVER_IDLE_TEXT)
        # A stale (x, y) from the just-closed map must not outlive it -- the
        # bounds check in paste_region()/copy_region() would likely catch a
        # mismatch against a differently-sized map opened next anyway, but
        # relying on that coincidence is exactly the kind of leak
        # edit_history.reset() above is already here to prevent for edit
        # history. The clipboard HISTORY deliberately survives a close, for
        # the same reason the single slot it replaced did (a
        # clipboard outliving the file it was copied from is normal
        # clipboard semantics, and Paste is already disabled with no map
        # loaded via _update_tool_enabled() below) -- self._region, unlike
        # the clipboard, IS map-relative (a stale tile rect indexing a map
        # that no longer exists), so it clears here alongside the hover
        # position. map_view.clear_image() above already dropped its own
        # rendering copy of the same value.
        self._hover_tile = None
        self._region = None
        self._update_tool_enabled()
        self._update_edit_actions()
        self._update_title()
        self._log_status(f"Closed {name}")
