"""The Clipboard History window -- pick which copied region Paste Region uses.

Dumb in exactly the way trigger_panel.VariablesDialog is: it never imports
ClipboardHistory, never touches EditHistory, and reports intent through the
callbacks it is constructed with. It is handed plain rows, not entry objects,
so the Qt half and the collection half stay independently testable.

Non-modal (show(), never exec_()) for two independent reasons. A modal dialog
freezes the window's _hover_tile, which is what Paste Region anchors on, so
paste would land wherever the mouse last was over the map. And exec_() hangs an
offscreen test run, which is why the other dialog openers are excluded from GUI
tests -- this one is testable precisely because it is non-modal.

A focused child QDialog is a separate top-level window, so the main window's
default-WindowShortcut actions do not fire here: Ctrl+V does nothing while this
has focus. That is correct and wanted, since a paste triggered from in here
would land on a stale anchor. The workflow is Set Active, click the map, hover,
paste, and the hint label says so. Qt.ApplicationShortcut would "fix" this into
a bug.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

_ICON_PX = 48


def _thumbnail_icon(thumbnail: np.ndarray) -> QPixmap | None:
    """Follows constant_picker.preview_pixmap()'s conversion exactly.
    QPixmap.fromImage() copies, so the local array going out of scope is safe.

    FastTransformation, not Smooth: smoothing blurs flat terrain colors into
    mud, which is the one thing this thumbnail exists to show."""
    if thumbnail is None or thumbnail.size == 0:
        return None
    array = np.ascontiguousarray(thumbnail)
    height, width = array.shape[:2]
    image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)
    return QPixmap.fromImage(image).scaled(
        _ICON_PX, _ICON_PX, Qt.KeepAspectRatio, Qt.FastTransformation
    )


class ClipboardHistoryDialog(QDialog):
    _EMPTY = "The clipboard is empty. Select a region and press Copy Region."
    _HINT = (
        "Paste anchors on the tile under the mouse - click the map, hover the "
        "target, then press Paste Region."
    )

    def __init__(self, parent=None, on_activate=None, on_delete=None, on_clear=None, on_rename=None):
        super().__init__(parent)
        self.setWindowTitle("Clipboard History")
        self.resize(460, 420)
        self._on_activate = on_activate or (lambda *args: None)
        self._on_delete = on_delete or (lambda *args: None)
        self._on_clear = on_clear or (lambda *args: None)
        self._on_rename = on_rename or (lambda *args: None)

        # A plain QTreeWidget, not trigger_panel's _HScrollStableTreeWidget:
        # that subclass exists to cancel Qt's horizontal auto-scroll toward a
        # wide ResizeToContents column, and here the only wide column is
        # Stretch, so the case never arises.
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["", "Name", "Size", "Units"])
        self.tree.setColumnCount(4)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setIconSize(QSize(_ICON_PX, _ICON_PX))
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.currentItemChanged.connect(lambda *_: self._update_buttons())
        self.tree.itemDoubleClicked.connect(lambda *_: self._request_activate())

        self.status = QLabel(self._EMPTY)
        self.status.setWordWrap(True)
        self.hint = QLabel(self._HINT)
        self.hint.setWordWrap(True)

        self.activate_button = QPushButton("Set Active")
        self.activate_button.setToolTip("Paste Region will use this entry (or double-click its row)")
        self.activate_button.clicked.connect(lambda checked=False: self._request_activate())
        self.rename_button = QPushButton("Rename")
        self.rename_button.setToolTip("Rename the selected entry")
        self.rename_button.clicked.connect(lambda checked=False: self._request_rename())
        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Remove the selected entry from the history")
        self.delete_button.clicked.connect(lambda checked=False: self._request_delete())
        self.clear_button = QPushButton("Clear All")
        self.clear_button.setToolTip("Empty the clipboard history")
        self.clear_button.clicked.connect(lambda checked=False: self._request_clear())

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addWidget(self.activate_button)
        button_row.addWidget(self.rename_button)
        button_row.addWidget(self.delete_button)
        button_row.addStretch(1)
        button_row.addWidget(self.clear_button)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tree, stretch=1)
        layout.addWidget(self.status)
        layout.addLayout(button_row)
        layout.addWidget(self.hint)
        layout.addWidget(buttons)

        self.set_entries([], None)

    def set_entries(self, rows, active_id: int | None) -> None:
        """Repopulate from `rows`, a sequence of
        (entry_id, label, width, height, unit_count, thumbnail_rgb).

        Selection is restored by entry id rather than by row, so a Ctrl+C on
        the main window while this is open does not move the user's highlight:
        a removal renumbers no id but does shift every later row."""
        selected = self.selected_entry_id()
        self.tree.clear()
        for entry_id, label, width, height, unit_count, thumbnail in rows:
            item = QTreeWidgetItem([" ", label or "", f"{width}x{height}", str(unit_count)])
            item.setData(0, Qt.UserRole, entry_id)
            pixmap = _thumbnail_icon(thumbnail)
            if pixmap is not None:
                item.setIcon(0, QIcon(pixmap))
            if entry_id == active_id:
                font = item.font(1)
                font.setBold(True)
                for column in range(self.tree.columnCount()):
                    item.setFont(column, font)
            self.tree.addTopLevelItem(item)
        active_label = next((r[1] for r in rows if r[0] == active_id), None)
        if not rows:
            self.status.setText(self._EMPTY)
        else:
            self.status.setText(f"Active: {active_label}" if active_label else "")
        self.status.setVisible(bool(self.status.text()))
        if selected is not None:
            self.select_entry(selected)
        self._update_buttons()

    def select_entry(self, entry_id: int) -> None:
        for row in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(row)
            if item.data(0, Qt.UserRole) == entry_id:
                self.tree.setCurrentItem(item)
                return

    def selected_entry_id(self) -> int | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.UserRole)
        return value if isinstance(value, int) else None

    def _update_buttons(self) -> None:
        has_selection = self.selected_entry_id() is not None
        self.activate_button.setEnabled(has_selection)
        self.rename_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)
        self.clear_button.setEnabled(self.tree.topLevelItemCount() > 0)

    # Each _request_* reads selected_entry_id() into a local BEFORE invoking
    # its callback and must not touch currentItem() afterwards: the window
    # repopulates this dialog from inside the callback. VariablesDialog's
    # _request_add documents the same race.

    def _request_activate(self) -> None:
        entry_id = self.selected_entry_id()
        if entry_id is None:
            return
        self._on_activate(entry_id)

    def _request_delete(self) -> None:
        entry_id = self.selected_entry_id()
        if entry_id is None:
            return
        self._on_delete(entry_id)

    def _request_clear(self) -> None:
        if self.tree.topLevelItemCount() == 0:
            return
        self._on_clear()

    def _request_rename(self) -> None:
        entry_id = self.selected_entry_id()
        if entry_id is None:
            return
        current = self.tree.currentItem().text(1)
        new_label, ok = QInputDialog.getText(
            self, "Rename Entry", "Name:", QLineEdit.Normal, current
        )
        if not ok:
            return
        new_label = new_label.strip()
        if not new_label or new_label == current:
            return
        self._on_rename(entry_id, new_label)
