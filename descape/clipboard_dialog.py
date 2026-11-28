"""The Clipboard History window: pick which copied region or units Paste uses,
from this session's copies or, on its Prefabs tab, the saved prefabs
(TASK-269), where using one pushes it as the new active entry.

Dumb in exactly the way trigger_panel_variables.VariablesDialog is: it never imports
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
from PyQt5.QtGui import QIcon, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
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
    _EMPTY = "The clipboard is empty. Select a region or units and press Copy."
    _HINT = (
        "Paste anchors on the tile under the mouse - click the map, hover the "
        "target, then press Paste."
    )

    _NO_PREFABS = "No prefabs saved yet. Copy a region or units, then press Save as Prefab."
    _PREFAB_HINT = "Use makes a prefab the active clipboard entry; then paste it as any copy."

    def __init__(
        self,
        parent=None,
        on_activate=None,
        on_delete=None,
        on_clear=None,
        on_rename=None,
        on_save_prefab=None,
        on_use_prefab=None,
        on_rename_prefab=None,
        on_delete_prefab=None,
        on_open_prefab_folder=None,
        on_prefabs_shown=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Clipboard History")
        self.resize(520, 440)
        noop = lambda *args: None  # noqa: E731
        self._on_activate = on_activate or noop
        self._on_delete = on_delete or noop
        self._on_clear = on_clear or noop
        self._on_rename = on_rename or noop
        self._on_save_prefab = on_save_prefab or noop
        self._on_use_prefab = on_use_prefab or noop
        self._on_rename_prefab = on_rename_prefab or noop
        self._on_delete_prefab = on_delete_prefab or noop
        self._on_open_prefab_folder = on_open_prefab_folder or noop
        self._on_prefabs_shown = on_prefabs_shown or noop

        # A plain QTreeWidget, not value_picker's _HScrollStableTreeWidget:
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
        self.activate_button.setToolTip("Paste will use this entry (or double-click its row)")
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
        self.save_prefab_button = QPushButton("Save as Prefab…")
        self.save_prefab_button.setToolTip("Save the selected entry as a prefab, kept between sessions")
        self.save_prefab_button.clicked.connect(lambda checked=False: self._request_save_prefab())

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addWidget(self.activate_button)
        button_row.addWidget(self.rename_button)
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.save_prefab_button)
        button_row.addStretch(1)
        button_row.addWidget(self.clear_button)

        session = QWidget()
        session_layout = QVBoxLayout(session)
        session_layout.addWidget(self.tree, stretch=1)
        session_layout.addWidget(self.status)
        session_layout.addLayout(button_row)
        session_layout.addWidget(self.hint)

        self.tabs = QTabWidget()
        self.tabs.addTab(session, "Session")
        self.tabs.addTab(self._build_prefabs_tab(), "Prefabs")
        self.tabs.currentChanged.connect(self._tab_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, stretch=1)
        layout.addWidget(buttons)

        self.set_entries([], None)
        self.set_prefab_rows([])

    def _build_prefabs_tab(self) -> QWidget:
        self.prefab_tree = QTreeWidget()
        self.prefab_tree.setHeaderLabels(["", "Name", "Size", "Units", "Saved"])
        self.prefab_tree.setRootIsDecorated(False)
        self.prefab_tree.setUniformRowHeights(True)
        self.prefab_tree.setIconSize(QSize(_ICON_PX, _ICON_PX))
        header = self.prefab_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for column in (2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.prefab_tree.currentItemChanged.connect(lambda *_: self._update_prefab_buttons())
        self.prefab_tree.itemDoubleClicked.connect(lambda *_: self._request_use_prefab())

        self.prefab_status = QLabel(self._NO_PREFABS)
        self.prefab_status.setWordWrap(True)
        prefab_hint = QLabel(self._PREFAB_HINT)
        prefab_hint.setWordWrap(True)

        self.use_prefab_button = QPushButton("Use")
        self.use_prefab_button.setToolTip("Make this prefab the active clipboard entry (or double-click its row)")
        self.use_prefab_button.clicked.connect(lambda checked=False: self._request_use_prefab())
        self.rename_prefab_button = QPushButton("Rename")
        self.rename_prefab_button.setToolTip("Rename the selected prefab")
        self.rename_prefab_button.clicked.connect(lambda checked=False: self._request_rename_prefab())
        self.delete_prefab_button = QPushButton("Delete")
        self.delete_prefab_button.setToolTip("Delete the selected prefab's file (this cannot be undone)")
        self.delete_prefab_button.clicked.connect(lambda checked=False: self._request_delete_prefab())
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.setToolTip("Show the prefab folder; copy files in or out of it to share prefabs")
        self.open_folder_button.clicked.connect(lambda checked=False: self._on_open_prefab_folder())

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for button in (self.use_prefab_button, self.rename_prefab_button, self.delete_prefab_button):
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(self.open_folder_button)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.prefab_tree, stretch=1)
        layout.addWidget(self.prefab_status)
        layout.addLayout(row)
        layout.addWidget(prefab_hint)
        return page

    def show_tab(self, which: str) -> None:
        """ "session" or "prefabs"."""
        self.tabs.setCurrentIndex(1 if which == "prefabs" else 0)

    def _tab_changed(self, index: int) -> None:
        if index == 1:
            self._on_prefabs_shown()

    def set_prefab_rows(self, rows) -> None:
        """Repopulate the Prefabs tab from `rows`, a sequence of (path, name,
        game_note, size_text, unit_count, saved_text, thumbnail_rgb, error).
        A row with an error is greyed, its reason the tooltip. Selection is
        restored by path."""
        selected = self.selected_prefab_path()
        self.prefab_tree.clear()
        for path, name, game_note, size_text, unit_count, saved_text, thumbnail, error in rows:
            label = f"{name} ({game_note})" if game_note else name
            units = "" if error else str(unit_count)
            item = QTreeWidgetItem([" ", label, "" if error else size_text, units, "" if error else saved_text])
            item.setData(0, Qt.UserRole, path)
            item.setData(1, Qt.UserRole, name)
            item.setData(2, Qt.UserRole, error)
            pixmap = None if error else _thumbnail_icon(thumbnail)
            if pixmap is not None:
                item.setIcon(0, QIcon(pixmap))
            if error:
                grey = self.palette().color(QPalette.Disabled, QPalette.Text)
                for column in range(self.prefab_tree.columnCount()):
                    item.setForeground(column, grey)
                    item.setToolTip(column, f"Can't be used: {error}")
            self.prefab_tree.addTopLevelItem(item)
        self.prefab_status.setText("" if rows else self._NO_PREFABS)
        self.prefab_status.setVisible(not rows)
        if selected is not None:
            self.select_prefab(selected)
        self._update_prefab_buttons()

    def select_prefab(self, path: str) -> None:
        for row in range(self.prefab_tree.topLevelItemCount()):
            item = self.prefab_tree.topLevelItem(row)
            if item.data(0, Qt.UserRole) == path:
                self.prefab_tree.setCurrentItem(item)
                return

    def selected_prefab_path(self) -> str | None:
        item = self.prefab_tree.currentItem()
        value = None if item is None else item.data(0, Qt.UserRole)
        return value if isinstance(value, str) else None

    def _selected_prefab_error(self) -> str | None:
        item = self.prefab_tree.currentItem()
        return None if item is None else item.data(2, Qt.UserRole)

    def _update_prefab_buttons(self) -> None:
        selected = self.selected_prefab_path() is not None
        readable = selected and not self._selected_prefab_error()
        self.use_prefab_button.setEnabled(readable)
        self.rename_prefab_button.setEnabled(readable)
        self.delete_prefab_button.setEnabled(selected)

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
        self.save_prefab_button.setEnabled(has_selection)
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

    def _request_save_prefab(self) -> None:
        entry_id = self.selected_entry_id()
        if entry_id is None:
            return
        self._on_save_prefab(entry_id)

    def _request_use_prefab(self) -> None:
        path = self.selected_prefab_path()
        if path is None or self._selected_prefab_error():
            return
        self._on_use_prefab(path)

    def _request_delete_prefab(self) -> None:
        path = self.selected_prefab_path()
        if path is None:
            return
        self._on_delete_prefab(path)

    def _request_rename_prefab(self) -> None:
        path = self.selected_prefab_path()
        if path is None or self._selected_prefab_error():
            return
        current = self.prefab_tree.currentItem().data(1, Qt.UserRole)
        new_name, ok = QInputDialog.getText(self, "Rename Prefab", "Name:", QLineEdit.Normal, current)
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == current:
            return
        self._on_rename_prefab(path, new_name)
