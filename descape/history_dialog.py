"""The Edit History window (GH #30) -- every recorded edit, newest last, with
"jump to this entry" as the one action.

Dumb in exactly the way clipboard_dialog.ClipboardHistoryDialog is: it never
imports EditHistory, is handed plain row tuples rather than DiffRecords, and
reports intent through the on_jump callback it is constructed with. That is
what keeps the Qt half and the history half independently testable.

Non-modal (show(), never exec_()) for the same two reasons the clipboard
dialog documents: a modal dialog would freeze the main window while the user
is trying to watch the map change under a jump, and exec_() hangs an offscreen
test run.

A cursor position, not a record index, is what a row carries: row 0 is the
synthetic "Opened file" entry standing for cursor 0, so the user can jump all
the way back to the freshly-loaded file, and record i sits at cursor i+1.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
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

# The synthetic first row's id. Real rows carry id(record), which CPython
# never hands out as 0, so this cannot collide.
OPENED_FILE_ID = 0
OPENED_FILE_LABEL = "Opened file"

_SAVED_MARKER = "✓"
_UNDONE_GREY = QColor(140, 140, 140)


class EditHistoryDialog(QDialog):
    _EMPTY = "Nothing has been edited yet."
    _HINT = "Double-click an entry to undo or redo up to it."

    def __init__(self, parent=None, on_jump=None):
        super().__init__(parent)
        self.setWindowTitle("Edit History")
        self.resize(460, 420)
        self._on_jump = on_jump or (lambda *args: None)

        # A plain QTreeWidget for the same reason clipboard_dialog gives: the
        # only wide column here is Stretch, so trigger_panel's
        # horizontal-auto-scroll subclass has nothing to cancel.
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["", "Edit", "Changes"])
        self.tree.setColumnCount(3)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.currentItemChanged.connect(lambda *_: self._update_buttons())
        self.tree.itemDoubleClicked.connect(lambda *_: self._request_jump())

        self.status = QLabel(self._EMPTY)
        self.status.setWordWrap(True)
        self.hint = QLabel(self._HINT)
        self.hint.setWordWrap(True)

        self.jump_button = QPushButton("Jump Here")
        self.jump_button.setToolTip("Undo or redo up to the selected entry (or double-click its row)")
        self.jump_button.clicked.connect(lambda checked=False: self._request_jump())

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addWidget(self.jump_button)
        button_row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tree, stretch=1)
        layout.addWidget(self.status)
        layout.addLayout(button_row)
        layout.addWidget(self.hint)
        layout.addWidget(buttons)

        self.set_rows([], 0, 0)

    def set_rows(self, rows, cursor: int, saved_at: int | None) -> None:
        """Repopulate from `rows`, a sequence of (row_id, label, kinds_text)
        in history order (oldest first), one per DiffRecord. `cursor` and
        `saved_at` are cursor positions in EditHistory's own numbering.

        Selection is restored by row id rather than by row number: _push()'s
        redo truncation and its overflow trim both shift every later index,
        the same reason ClipboardHistoryDialog.set_entries gives."""
        selected = self.selected_row_id()
        self.tree.clear()
        entries = [(OPENED_FILE_ID, OPENED_FILE_LABEL, "")]
        entries.extend((row_id, label, kinds_text) for row_id, label, kinds_text in rows)
        current_item = None
        for target, (row_id, label, kinds_text) in enumerate(entries):
            marker = _SAVED_MARKER if saved_at is not None and target == saved_at else ""
            item = QTreeWidgetItem([marker, label or "", kinds_text or ""])
            item.setData(0, Qt.UserRole, row_id)
            item.setData(1, Qt.UserRole, target)
            if marker:
                item.setToolTip(0, "The state last saved to disk")
            if target == cursor:
                font = item.font(1)
                font.setBold(True)
                for column in range(self.tree.columnCount()):
                    item.setFont(column, font)
                current_item = item
            elif target > cursor:
                # Undone: still replayable by jumping forward, so greyed
                # rather than dropped.
                for column in range(self.tree.columnCount()):
                    item.setForeground(column, _UNDONE_GREY)
            self.tree.addTopLevelItem(item)
        if not rows:
            self.status.setText(self._EMPTY)
        else:
            current_label = entries[cursor][1] if 0 <= cursor < len(entries) else ""
            self.status.setText(f"Current: {current_label} ({len(rows)} edits)")
        if selected is not None:
            self.select_row(selected)
        if current_item is not None:
            self.tree.scrollToItem(current_item)
        self._update_buttons()

    def select_row(self, row_id: int) -> None:
        for row in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(row)
            if item.data(0, Qt.UserRole) == row_id:
                self.tree.setCurrentItem(item)
                return

    def selected_row_id(self) -> int | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.UserRole)
        return value if isinstance(value, int) else None

    def selected_target(self) -> int | None:
        """The cursor position the selected row stands for -- stored on the
        item rather than derived from its row number, so a caller never has
        to know the synthetic first row's off-by-one."""
        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(1, Qt.UserRole)
        return value if isinstance(value, int) else None

    def _update_buttons(self) -> None:
        self.jump_button.setEnabled(self.selected_target() is not None)

    def _request_jump(self) -> None:
        # Reads the target into a local BEFORE invoking the callback and must
        # not touch currentItem() afterwards: the jump repopulates this dialog
        # from inside that callback (EditHistory.on_change). Same race
        # ClipboardHistoryDialog's _request_* block documents.
        target = self.selected_target()
        if target is None:
            return
        self._on_jump(target)
