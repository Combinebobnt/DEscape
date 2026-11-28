"""File > Browse Game Scenarios…: one filterable list of the scenarios in
every profile id's and mod's scenario folder, disabled mods included.

Dumb in the same way autosave_dialog.RecoverAutosaveDialog is: it lists what
scenario_library.discover() finds and reports a pick through on_open(path).
It never loads a scenario itself. Modeless (show(), never exec_(), which hangs
an offscreen test run) and refreshed on every show.
"""

from __future__ import annotations

from datetime import UTC, datetime

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from descape import content_roots, item_activation, scenario_library
from descape.autosave_dialog import size_text

COL_NAME, COL_SOURCE, COL_MODIFIED, COL_SIZE = range(4)
SORT_ROLE = Qt.UserRole + 1
NO_PROFILE_TEXT = "No AoE2:DE profile folder found. Set one in Settings > Mods."
NO_FILES_TEXT = "No scenarios in the profile or mod folders."


def _live_source():
    """(entries, whether a profile folder was found)."""
    return scenario_library.discover(), content_roots.profile_path() is not None


class _Item(QTreeWidgetItem):
    def __lt__(self, other) -> bool:
        column = self.treeWidget().sortColumn() if self.treeWidget() else COL_NAME
        return self.data(column, SORT_ROLE) < other.data(column, SORT_ROLE)


class ScenarioBrowserDialog(QDialog):
    def __init__(self, parent, on_open, source=_live_source):
        super().__init__(parent)
        self.setWindowTitle("Browse Game Scenarios")
        self.resize(760, 420)
        self._on_open = on_open
        self._source = source
        self._total = 0
        self._has_profile = True

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Scenarios in the game's profile and mod folders. One from a subscribed mod "
            "opens as usual, but Save asks for a new name, since the game re-syncs that folder. "
            "Older .scn, .scx and .scx2 files open for viewing only."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        filter_row = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by name or source")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_edit, 1)
        self.count_label = QLabel()
        filter_row.addWidget(self.count_label)
        layout.addLayout(filter_row)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Name", "Source", "Modified", "Size"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        for column in (COL_SOURCE, COL_MODIFIED, COL_SIZE):
            self.tree.header().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(COL_MODIFIED, Qt.DescendingOrder)  # a scenario just saved in-game is on top
        self.tree.itemActivated.connect(lambda item, _column: self._open_item(item))
        self.tree.installEventFilter(self)
        layout.addWidget(self.tree)

        buttons_row = QHBoxLayout()
        self.open_button = QPushButton("Open")
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(lambda: self._open_item(self.tree.currentItem()))
        buttons_row.addWidget(self.open_button)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setAutoDefault(False)
        self.refresh_button.clicked.connect(self.refresh)
        buttons_row.addWidget(self.refresh_button)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.rejected.connect(self.close)
        close_box.button(QDialogButtonBox.Close).clicked.connect(self.close)
        close_box.button(QDialogButtonBox.Close).setAutoDefault(False)
        layout.addWidget(close_box)

        self.refresh()

    def eventFilter(self, watched, event) -> bool:
        """Return on a row opens it once: activating leaves the key ignored, so QDialog would press Open too."""
        if watched is self.tree and event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            item_activation.activate_current(self.tree)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def refresh(self) -> None:
        entries, self._has_profile = self._source()
        self.tree.setSortingEnabled(False)  # one sort after the fill, not one per row
        self.tree.clear()
        for entry in entries:
            item = _Item([
                entry.name,
                entry.source,
                datetime.fromtimestamp(entry.mtime, UTC).astimezone().strftime("%Y-%m-%d %H:%M"),
                size_text(entry.size),
            ])
            item.setData(COL_NAME, Qt.UserRole, entry.path)
            for column, key in (
                (COL_NAME, entry.name.casefold()),
                (COL_SOURCE, entry.source.casefold()),
                (COL_MODIFIED, entry.mtime),
                (COL_SIZE, entry.size),
            ):
                item.setData(column, SORT_ROLE, key)
            for column in range(4):
                item.setToolTip(column, str(entry.path))
            self.tree.addTopLevelItem(item)
        self._total = len(entries)
        self.tree.setSortingEnabled(True)
        self._apply_filter()

    def visible_items(self) -> list[QTreeWidgetItem]:
        items = (self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount()))
        return [item for item in items if not item.isHidden()]

    def _apply_filter(self) -> None:
        needle = self.filter_edit.text().strip().casefold()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            haystack = f"{item.text(COL_NAME)}\n{item.text(COL_SOURCE)}".casefold()
            item.setHidden(bool(needle) and needle not in haystack)
        shown = self.visible_items()
        if not self._has_profile and not self._total:
            self.count_label.setText(NO_PROFILE_TEXT)
        elif not self._total:
            self.count_label.setText(NO_FILES_TEXT)
        else:
            noun = "scenario" if self._total == 1 else "scenarios"
            shown_of = f"{len(shown)} of " if needle else ""
            self.count_label.setText(f"{shown_of}{self._total} {noun}")
        self.open_button.setEnabled(bool(shown))
        current = self.tree.currentItem()
        if shown and (current is None or current.isHidden()):
            self.tree.setCurrentItem(shown[0])

    def _open_item(self, item) -> None:
        if item is not None and not item.isHidden():
            self._on_open(item.data(COL_NAME, Qt.UserRole))
