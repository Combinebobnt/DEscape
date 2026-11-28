"""ViewerWindow's map prefabs (TASK-269): Save as Prefab, the Prefabs tab's
rows, and Use, Rename, Delete and Open Folder. The file format is
descape/prefabs.py. Use pushes the prefab's block as the new active
clipboard entry, so paste and its refusals are Copy's own."""

from __future__ import annotations

import datetime
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QInputDialog, QLineEdit, QMessageBox

from descape import debug_log, games, object_catalog, prefabs


def _saved_text(saved: str) -> str:
    """The header's UTC ISO time as local "YYYY-MM-DD HH:MM", or the text itself if it doesn't parse."""
    try:
        when = datetime.datetime.fromisoformat(saved)
    except ValueError:
        return saved
    if when.tzinfo is not None:
        when = when.astimezone()
    return when.strftime("%Y-%m-%d %H:%M")


def _size_text(info: prefabs.PrefabInfo) -> str:
    return f"{info.width}x{info.height}" if info.has_tiles else "units only"


class _ViewerPrefabsMixin:
    """ViewerWindow's map prefabs (viewer.py's docstring has the split's rules)."""

    def _prefab_dialog_callbacks(self) -> dict:
        return {
            "on_save_prefab": self._save_entry_as_prefab,
            "on_use_prefab": self._use_prefab,
            "on_rename_prefab": self._rename_prefab,
            "on_delete_prefab": self._delete_prefab,
            "on_open_prefab_folder": self._open_prefab_folder,
            "on_prefabs_shown": self._refresh_prefab_rows,
        }

    def _show_prefabs(self) -> None:
        self._show_clipboard_history(tab="prefabs")

    def _prefab_game_note(self, info: prefabs.PrefabInfo) -> str:
        """What Paste would refuse the prefab for here: its game when it isn't the open file's own
        (the default game with none open), and an older format on a DE document; else ""."""
        notes = []
        if info.dataset != self._file_game_dataset():
            notes.append(games.get(info.dataset).display_name)
        if info.legacy and not self._file_is_legacy():
            notes.append("older format")
        return ", ".join(notes)

    def _prefab_rows(self) -> list[tuple]:
        rows = []
        for info in prefabs.list_prefabs():
            if info.error is not None:
                rows.append((str(info.path), info.name, "", "", 0, "", None, info.error))
                continue
            rows.append(
                (
                    str(info.path),
                    info.name,
                    self._prefab_game_note(info),
                    _size_text(info),
                    info.unit_count,
                    _saved_text(info.saved),
                    info.thumbnail,
                    None,
                )
            )
        return rows

    def _refresh_prefab_rows(self) -> None:
        if self._clipboard_dialog is None:
            return
        self._clipboard_dialog.set_prefab_rows(self._prefab_rows())

    def _prefab_prompt_parent(self):
        """The Clipboard History window while it is shown (its buttons ask), else the main window."""
        dialog = self._clipboard_dialog
        return dialog if dialog is not None and dialog.isVisible() else self

    def _ask_prefab_name(self, default: str) -> str | None:
        name, ok = QInputDialog.getText(self._prefab_prompt_parent(), "Save as Prefab", "Prefab name:", QLineEdit.Normal, default)
        return name.strip() if ok else None

    def _confirm_prefab(self, title: str, text: str) -> bool:
        reply = QMessageBox.question(self._prefab_prompt_parent(), title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply == QMessageBox.Yes

    def _save_entry_as_prefab(self, entry_id: int | None = None) -> None:
        """Saves a clipboard entry, the active one by default (Edit > Save as Prefab…)."""
        history = self._clipboard_history
        entry_id = history.active_id if entry_id is None else entry_id
        entry = next((e for e in history.entries if e.entry_id == entry_id), None)
        if entry is None:
            self._log_status("Save as Prefab: the clipboard is empty")
            return
        name = self._ask_prefab_name(entry.label)
        if not name:
            return
        try:
            try:
                prefabs.save_block(entry.block, name)
            except prefabs.PrefabNameTaken as taken:
                if not self._confirm_prefab("Replace Prefab", f'A prefab named "{name}" already exists. Replace it?'):
                    return
                prefabs.save_block(entry.block, name, replace_path=taken.path)
        except prefabs.PrefabRefused as exc:
            self._log_status(f"Save as Prefab refused: {exc}", level=debug_log.WARNING)
            return
        self._refresh_prefab_rows()
        self._log_status(f"Saved prefab {name}")

    def _unknown_object_count(self, block) -> int:
        """Units whose const this game's data doesn't define; 0 when Paste would refuse the block or the document anyway."""
        if self.scenario is None or self._cross_game_refusal(block) or self.game_data_lock_reason():
            return 0
        return sum(1 for unit in block.units if not object_catalog.is_known_object(unit.unit_const))

    def _use_prefab(self, path: str) -> None:
        try:
            info, block = prefabs.load(Path(path))
        except prefabs.PrefabRefused as exc:
            self._log_status(f"Prefab refused: {exc}", level=debug_log.WARNING)
            self._refresh_prefab_rows()
            return
        self._push_clipboard(block, label=info.name, restamp=False)
        message = f"Prefab {info.name} is the active clipboard entry: hover the map and Paste"
        unknown = self._unknown_object_count(block)
        if unknown:
            plural = "unit uses an object" if unknown == 1 else "units use objects"
            message += f". {unknown} {plural} this game data does not define, such as a data mod's or a newer patch's"
        self._log_status(message, level=debug_log.WARNING if unknown else debug_log.INFO)

    def _rename_prefab(self, path: str, name: str) -> None:
        try:
            prefabs.rename(Path(path), name)
        except prefabs.PrefabRefused as exc:
            self._log_status(f"Rename refused: {exc}", level=debug_log.WARNING)
            return
        finally:
            self._refresh_prefab_rows()
        self._log_status(f"Renamed prefab to {name.strip()}")

    def _delete_prefab(self, path: str) -> None:
        try:
            name = prefabs.read_info(Path(path)).name
        except prefabs.PrefabRefused:
            name = Path(path).stem
        if not self._confirm_prefab("Delete Prefab", f'Delete the prefab "{name}"? Its file is removed for good.'):
            return
        try:
            prefabs.delete(Path(path))
        except prefabs.PrefabRefused as exc:
            self._log_status(f"Delete refused: {exc}", level=debug_log.WARNING)
            return
        finally:
            self._refresh_prefab_rows()
        self._log_status(f"Deleted prefab {name}")

    def _open_prefab_folder(self) -> None:
        try:
            folder = prefabs.ensure_dir()
        except prefabs.PrefabRefused as exc:
            self._log_status(f"Open Folder refused: {exc}", level=debug_log.WARNING)
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._log_status(f"Could not open the prefab folder: {folder}", level=debug_log.WARNING)
