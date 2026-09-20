"""File > Recover from Autosave... -- pick a recovery slot and open it.

Its own module rather than another class in viewer.py, to keep the conflict
surface of the autosave feature small; only the timer, the tick and the
Settings tab genuinely have to live in the window.

Dumb in the same way clipboard_dialog.ClipboardHistoryDialog is: it lists
whatever autosave.entries() hands it and reports intent through the
callbacks it was constructed with. It never loads a scenario itself.

Non-modal (show(), never exec_()): exec_() hangs an offscreen test run,
which is what makes the other dialog openers untestable.
"""

from __future__ import annotations

import contextlib
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from descape import autosave


def _size_text(size: int) -> str:
    return f"{size / 1024:.0f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"


class RecoverAutosaveDialog(QDialog):
    def __init__(self, parent, on_open, on_delete=None):
        super().__init__(parent)
        self.setWindowTitle("Recover from Autosave")
        self.resize(640, 320)
        self._on_open = on_open
        self._on_delete = on_delete

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Autosaves open as a new untitled document, so recovering one "
            "can never overwrite the file you are comparing it against."
        ))

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Original", "Saved at", "Size"])
        self.tree.setRootIsDecorated(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.tree)

        buttons_row = QHBoxLayout()
        self.open_button = QPushButton("Open as New Document")
        self.open_button.clicked.connect(self._open_selected)
        buttons_row.addWidget(self.open_button)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_selected)
        buttons_row.addWidget(self.delete_button)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.rejected.connect(self.close)
        close_box.button(QDialogButtonBox.Close).clicked.connect(self.close)
        layout.addWidget(close_box)

        self.refresh()

    def refresh(self) -> None:
        self.tree.clear()
        for entry in autosave.entries():
            label = "Untitled" if entry.untitled else entry.display_path
            item = QTreeWidgetItem([
                label,
                datetime.fromtimestamp(entry.saved_at).strftime("%Y-%m-%d %H:%M:%S"),
                _size_text(entry.size),
            ])
            item.setData(0, Qt.UserRole, entry)
            self.tree.addTopLevelItem(item)
        empty = self.tree.topLevelItemCount() == 0
        self.open_button.setEnabled(not empty)
        self.delete_button.setEnabled(not empty)
        if not empty:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def selected_entry(self):
        item = self.tree.currentItem()
        return None if item is None else item.data(0, Qt.UserRole)

    def _open_selected(self) -> None:
        entry = self.selected_entry()
        if entry is not None:
            self._on_open(entry)

    def _delete_selected(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        if self._on_delete is not None:
            self._on_delete(entry)
        else:
            with contextlib.suppress(OSError):
                entry.path.unlink()
        self.refresh()
