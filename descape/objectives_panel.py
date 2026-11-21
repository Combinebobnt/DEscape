"""GH #391: the Objectives dock's widget. Every objective, header and
on-screen trigger in game order, with inline flags, order and short
description, drag and button reordering, and the game-style preview.

Passive like MessagesPanel: it never touches TriggerEditModel or
EditHistory. Each action reports through a callback and the window decides
what it means. Reorders are worked out here with objectives_model's pure
functions, so the window receives finished {trigger index: order} changes.

Anything the tree itself started (a checkbox, an inline edit, a drop) is
reported on a zero-delay timer: the window's write repopulates the tree, which
must not happen inside the item or editor signal that asked for it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence

from PyQt5.QtCore import QEvent, QItemSelectionModel, QRegularExpression, QSignalBlocker, Qt, QTimer
from PyQt5.QtGui import QColor, QRegularExpressionValidator
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from descape import asset_source, ui_icons
from descape.objectives_box import (
    NOTE,
    VARIABLE_BOTH,
    VARIABLE_NAME,
    VARIABLE_VALUE,
    BoxEntry,
    ObjectivesBox,
    ObjectivesOverlay,
)
from descape.objectives_model import (
    MAX_ORDER,
    ObjectiveRow,
    box_lines,
    has_string_id,
    moved_by_step,
    orders_after_move,
    renumbered,
)

COL_ORDER, COL_HEADER, COL_OBJECTIVE, COL_SCREEN, COL_MUTE, COL_PREVIEW, COL_SHORT, COL_TRIGGER = range(8)
# The eye column's header icon, from icon_art's artwork.
PREVIEW_ICON = "filter-show-all"
# (column, header label, trigger attribute, tooltip)
FLAG_COLUMNS = (
    (COL_HEADER, "H", "header", "Header: shown as a heading, with no checkbox"),
    (COL_OBJECTIVE, "Obj", "display_as_objective", "Display as Objective: listed in the game's Objectives panel"),
    (COL_SCREEN, "Scr", "display_on_screen", "Display on Screen: shown in the on-screen objectives box"),
    (COL_MUTE, "Mute", "mute_objectives", "Mute Objectives: no sound when it changes"),
)
_FLAG_ATTR = {column: attr for column, _label, attr, _tip in FLAG_COLUMNS}
_INDEX_ROLE = Qt.UserRole
_RAW_ROLE = Qt.UserRole + 1

PLACE_DOCK, PLACE_MAP, PLACE_OFF = "dock", "map", "off"
PLACES = ((PLACE_DOCK, "In dock"), (PLACE_MAP, "On map"), (PLACE_OFF, "Off"))
VARIABLE_MODES = ((VARIABLE_NAME, "Name"), (VARIABLE_VALUE, "Value"), (VARIABLE_BOTH, "Both"))

DISABLED_TIP = "Off at game start: the game shows it once a trigger activates it"
TICK_NOTE = "Click a checkbox to preview it ticked; right-click a row to hide it from the preview (neither is saved)."
HIDDEN_TIP = "Hidden from the preview only: the game still shows it. Tick its eye column to show it again"
PREVIEW_TIP = "Shown in the preview. Untick to leave it out of the box, here and on the map; never saved"
NO_DOCUMENT = "No map open."
NO_TRIGGERS = "This file's triggers cannot be read."
EMPTY_LIST = "No objectives yet. New Objective, New Header or Add Trigger… makes one."


def resolved_text(row: ObjectiveRow) -> tuple[str, str]:
    """(text the game shows, tooltip note) for `row`'s short description."""
    if not has_string_id(row.short_stid):
        return row.short, ""
    text = asset_source.resource_string(row.short_stid)
    if text is None:
        return row.short, f"Shows language string #{row.short_stid} (not available), else this text."
    return text, f"Text from language string #{row.short_stid}."


def box_entries(
    rows: Sequence[ObjectiveRow], at_start: bool, hidden: Collection[int] = (), typing: Mapping[int, str] | None = None
) -> list[BoxEntry]:
    """The box's lines; `typing` is short description text typed in Trigger Details but not committed yet."""
    typing = typing or {}
    rows = [row._replace(short=typing[row.index]) if row.index in typing else row for row in rows]
    entries = []
    for line in box_lines(rows, at_start):
        if line.row.index in hidden:
            continue
        text, note = resolved_text(line.row)
        entries.append(BoxEntry(line.kind, text, line.row.index, note))
    if not entries and rows:
        entries.append(BoxEntry(NOTE, "(nothing on screen at game start)" if at_start else "(nothing on screen)"))
    return entries


class _Delegate(QStyledItemDelegate):
    """Order as digits up to the field's u32 limit (wider than a QSpinBox),
    the short description as one line of its stored text (a string id's
    resolved text is only what the row shows)."""

    def __init__(self, panel: ObjectivesPanel) -> None:
        super().__init__(panel.tree)
        self._panel = panel

    def editorEvent(self, event, model, option, index) -> bool:
        # A read-only file's flags show but do not toggle: only a left click on the box itself or Space is
        # held back, so a click elsewhere in the cell still selects the row. The eye column always toggles.
        if index.column() in _FLAG_ATTR and not self._panel.editable():
            if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
                style_option = QStyleOptionViewItem(option)
                self.initStyleOption(style_option, index)
                widget = option.widget
                style = widget.style() if widget is not None else QApplication.style()
                box = style.subElementRect(QStyle.SE_ItemViewItemCheckIndicator, style_option, widget)
                if event.button() == Qt.LeftButton and box.contains(event.pos()):
                    return True
            elif event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Space, Qt.Key_Select):
                return True
        return super().editorEvent(event, model, option, index)

    def createEditor(self, parent, option, index):
        if not self._panel.editable():
            return None
        if index.column() == COL_ORDER:
            editor = QLineEdit(parent)
            editor.setObjectName("order_editor")
            editor.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{1,10}"), editor))
            return editor
        if index.column() == COL_SHORT:
            if "\n" in (index.data(_RAW_ROLE) or ""):
                return None
            return QLineEdit(parent)
        return None

    def setEditorData(self, editor, index) -> None:
        if isinstance(editor, QLineEdit):
            editor.setText(str(index.data(_RAW_ROLE) or (0 if index.column() == COL_ORDER else "")))

    def setModelData(self, editor, _model, index) -> None:
        trigger = index.siblingAtColumn(COL_ORDER).data(_INDEX_ROLE)
        if index.column() == COL_ORDER:
            text = editor.text()
            if text and int(text) <= MAX_ORDER and int(text) != int(index.data(_RAW_ROLE) or 0):
                self._panel.later(self._panel.on_order, {trigger: int(text)}, "Set objective order")
        elif editor.text() != (index.data(_RAW_ROLE) or ""):
            self._panel.later(self._panel.on_short, trigger, editor.text())


class _Tree(QTreeWidget):
    """Internal-move drags whose drop reports the target row instead of moving items."""

    def __init__(self, panel: ObjectivesPanel) -> None:
        super().__init__()
        self._panel = panel

    def dropEvent(self, event) -> None:
        if event.source() is not self:
            event.ignore()
            return
        item = self.itemAt(event.pos())
        if item is None:
            position = self.topLevelItemCount()
        else:
            position = self.indexOfTopLevelItem(item)
            if event.pos().y() > self.visualItemRect(item).center().y():
                position += 1
        # Nothing is moved here: the window rewrites the orders and repopulates.
        event.setDropAction(Qt.IgnoreAction)
        event.accept()
        self._panel.drop_rows(self._panel.selected_indices(), position)


class _PickTriggers(QDialog):
    """Add Trigger…'s picker: every trigger that is not an objective yet, filtered by name."""

    def __init__(self, parent: QWidget, candidates: Sequence[tuple[int, str]]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Trigger as Objective")
        layout = QVBoxLayout(self)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter triggers…")
        self.filter_edit.setClearButtonEnabled(True)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for index, label in candidates:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, index)
            self.list.addItem(item)
        self.filter_edit.textChanged.connect(self._filter)
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Make Objective")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(QLabel("Triggers to show as objectives (Display as Objective and on Screen):"))
        layout.addWidget(self.filter_edit)
        layout.addWidget(self.list)
        layout.addWidget(buttons)
        self.resize(420, 480)

    def _filter(self, text: str) -> None:
        needle = text.casefold()
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setHidden(needle not in item.text().casefold())

    def picked(self) -> list[int]:
        items = sorted((item for item in self.list.selectedItems() if not item.isHidden()), key=self.list.row)
        return [item.data(Qt.UserRole) for item in items]


class ObjectivesPanel(QWidget):
    """The list, its buttons, the preview controls and the in-dock box."""

    def __init__(
        self,
        on_flags: Callable[[list[int], str, bool], None] | None = None,
        on_order: Callable[[dict[int, int], str], None] | None = None,
        on_short: Callable[[int, str], None] | None = None,
        on_new: Callable[[str, int | None], None] | None = None,
        on_promote: Callable[[list[int]], None] | None = None,
        on_remove: Callable[[list[int]], None] | None = None,
        on_select: Callable[[int], None] | None = None,
        on_preview_place: Callable[[str], None] | None = None,
        on_at_start: Callable[[bool], None] | None = None,
        on_variable_mode: Callable[[str], None] | None = None,
        on_columns: Callable[[bytes], None] | None = None,
        on_clear_string_ids: Callable[[list[int]], None] | None = None,
    ) -> None:
        super().__init__()
        noop = lambda *args: None  # noqa: E731
        self.on_flags = on_flags or noop
        self.on_order = on_order or noop
        self.on_short = on_short or noop
        self.on_new = on_new or noop
        self.on_promote = on_promote or noop
        self.on_remove = on_remove or noop
        self.on_select = on_select or noop
        self.on_preview_place = on_preview_place or noop
        self.on_at_start = on_at_start or noop
        self.on_variable_mode = on_variable_mode or noop
        self.on_columns = on_columns or noop
        self.on_clear_string_ids = on_clear_string_ids or noop
        # The window's (trigger index, label) list for Add Trigger…'s picker.
        self.candidates: Callable[[], list[tuple[int, str]]] = list
        self._rows: list[ObjectiveRow] = []
        self._editable = False
        self._populating = False
        self._ticked: set[int] = set()
        # Rows left out of the preview this session, by trigger index; never saved.
        self._hidden: set[int] = set()
        # Short descriptions typed in Trigger Details and not committed yet, by trigger index.
        self._typing: dict[int, str] = {}
        self._at_start = False
        self._place = PLACE_DOCK
        self._variables: tuple[str, ...] = ()
        self._variable_mode = VARIABLE_NAME
        self._overlay: ObjectivesOverlay | None = None
        self._pending: list[tuple[Callable, tuple]] = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(0)
        self._flush_timer.timeout.connect(self.flush)
        self._build()

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        self.tree = _Tree(self)
        tree = self.tree
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        tree.setUniformRowHeights(True)
        tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        tree.setDragDropMode(QAbstractItemView.InternalMove)
        tree.setDefaultDropAction(Qt.MoveAction)
        tree.setItemDelegate(_Delegate(self))
        tree.setHeaderLabels(["Order", *(label for _c, label, _a, _t in FLAG_COLUMNS), "", "Short description", "Trigger"])
        tree.headerItem().setIcon(COL_PREVIEW, ui_icons.icon(PREVIEW_ICON))
        tree.headerItem().setToolTip(COL_PREVIEW, "Shown in the preview (not saved): untick a row to leave it out of the box")
        tree.headerItem().setToolTip(COL_ORDER, "Description Order: higher is listed first. Double-click to type one")
        for column, _label, _attr, tip in FLAG_COLUMNS:
            tree.headerItem().setToolTip(column, tip)
        tree.headerItem().setToolTip(COL_SHORT, "What the game shows. Double-click to edit; the long Description is in Trigger Details")
        header = tree.header()
        # The last column fills the rest, so every resize handle sits between two columns.
        header.setStretchLastSection(True)
        header.setSectionsMovable(True)
        header.setMinimumSectionSize(20)
        metrics = header.fontMetrics()
        for column, label in enumerate(["Order", *(label for _c, label, _a, _t in FLAG_COLUMNS)]):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            tree.setColumnWidth(column, metrics.horizontalAdvance(label) + (24 if column == COL_ORDER else 16))
        header.setSectionResizeMode(COL_PREVIEW, QHeaderView.Interactive)
        tree.setColumnWidth(COL_PREVIEW, metrics.horizontalAdvance("Mute") + 16)
        header.setSectionResizeMode(COL_SHORT, QHeaderView.Interactive)
        header.setSectionResizeMode(COL_TRIGGER, QHeaderView.Interactive)
        tree.setColumnWidth(COL_SHORT, metrics.horizontalAdvance("M") * 20)
        tree.setColumnWidth(COL_TRIGGER, metrics.horizontalAdvance("M") * 16)
        # A drag that reorders or resizes columns is saved once it settles.
        self._columns_timer = QTimer(self)
        self._columns_timer.setSingleShot(True)
        self._columns_timer.setInterval(500)
        self._columns_timer.timeout.connect(self._save_columns)
        self._saved_columns = bytes(header.saveState())
        # What changed since the last save: a move, or the logical columns resized. Judged in the timer.
        self._columns_moved = False
        self._columns_resized: set[int] = set()
        header.sectionMoved.connect(self._on_section_moved)
        header.sectionResized.connect(self._on_section_resized)
        tree.itemChanged.connect(self._on_item_changed)
        tree.currentItemChanged.connect(self._on_current_changed)
        tree.itemSelectionChanged.connect(self._update_buttons)
        tree.setContextMenuPolicy(Qt.CustomContextMenu)
        tree.customContextMenuRequested.connect(self._show_context_menu)

        self.note_label = QLabel(NO_DOCUMENT)
        self.note_label.setWordWrap(True)
        self.new_objective_button = self._button(
            "New Objective", "A new trigger shown as an objective and on screen, under the selected row"
        )
        self.new_header_button = self._button("New Header", "A new header trigger under the selected row")
        self.add_trigger_button = self._button("Add Trigger…", "Show existing triggers as objectives")
        self.remove_button = self._button("Remove", "Clear the selected rows' objective, header and on-screen flags. The triggers stay")
        self.up_button = self._button("▲", "Move the selected rows up one place")
        self.down_button = self._button("▼", "Move the selected rows down one place")
        self.renumber_button = self._button("Renumber", "Count Description Order down by 1 in the listed order, keeping the top value. "
            "With several rows selected, only those count down, from the first one's value")
        for button in (self.up_button, self.down_button):
            button.setFixedWidth(button.fontMetrics().horizontalAdvance("▲") + 24)
        self.new_objective_button.clicked.connect(lambda: self.on_new("objective", self.current_index()))
        self.new_header_button.clicked.connect(lambda: self.on_new("header", self.current_index()))
        self.add_trigger_button.clicked.connect(self._add_trigger)
        self.remove_button.clicked.connect(lambda: self.on_remove(self.selected_indices()))
        self.up_button.clicked.connect(lambda: self.move_selected(-1))
        self.down_button.clicked.connect(lambda: self.move_selected(1))
        self.renumber_button.clicked.connect(
            lambda: self.on_order(renumbered(self._rows, self.selected_indices()), "Renumber objectives")
        )
        buttons = QHBoxLayout()
        for button in (
            self.new_objective_button, self.new_header_button, self.add_trigger_button, self.remove_button,
            self.up_button, self.down_button, self.renumber_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch()

        self.place_combo = QComboBox()
        for place, label in PLACES:
            self.place_combo.addItem(label, place)
        self.place_combo.setToolTip(
            "Where the game-style box is drawn: in this panel, over the map's top-right corner as in the game, or nowhere"
        )
        self.place_combo.currentIndexChanged.connect(self._on_place_changed)
        self.all_radio = QRadioButton("All objectives")
        self.all_radio.setToolTip("Every objective shown on screen")
        self.start_radio = QRadioButton("At game start")
        self.start_radio.setToolTip("Only triggers enabled at game start, as the box first appears")
        self.all_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.all_radio)
        group.addButton(self.start_radio)
        self.start_radio.toggled.connect(self._on_at_start_toggled)
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Preview:"))
        preview_row.addWidget(self.place_combo)
        preview_row.addWidget(self.all_radio)
        preview_row.addWidget(self.start_radio)
        self.variable_combo = QComboBox()
        for mode, label in VARIABLE_MODES:
            self.variable_combo.addItem(label, mode)
        self.variable_combo.setToolTip(
            "How a <name> tag naming a trigger variable reads in the preview: as typed (as the game draws it), "
            "its value at game start (0), or both"
        )
        self.variable_combo.currentIndexChanged.connect(self._on_variable_mode_changed)
        preview_row.addWidget(QLabel("Variables:"))
        preview_row.addWidget(self.variable_combo)
        preview_row.addStretch()

        self.box = ObjectivesBox()
        self.box.state.on_tick = self.toggle_tick
        self.tick_note = QLabel(TICK_NOTE)
        self.tick_note.setEnabled(False)
        self.tick_note.setWordWrap(True)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(preview_row)
        left_layout.addWidget(self.note_label)
        left_layout.addWidget(self.tree, 1)
        left_layout.addLayout(buttons)
        self.preview = QWidget()
        right_layout = QVBoxLayout(self.preview)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.box_scroll = QScrollArea()
        self.box_scroll.setWidgetResizable(True)
        self.box_scroll.setFrameShape(QFrame.NoFrame)
        self.box_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.box_scroll.setWidget(self.box)
        right_layout.addWidget(self.box_scroll, 1)
        right_layout.addWidget(self.tick_note)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.preview)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.addWidget(self.splitter)
        # Scrolls rather than holding a minimum: tabbed under the map, its width would cap the side panel's.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self._update_buttons()

    def _button(self, text: str, tip: str) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tip)
        return button

    # -- content --------------------------------------------------------------

    def show_rows(self, rows: Sequence[ObjectiveRow], editable: bool, note: str = "") -> None:
        """Repopulate, keeping the selection, current row and scroll by trigger index."""
        selected = set(self.selected_indices())
        current = self.current_index()
        scroll = self.tree.verticalScrollBar().value()
        self._rows = list(rows)
        self._editable = bool(editable)
        self._ticked &= {row.index for row in self._rows}
        self._hidden &= {row.index for row in self._rows}
        self._typing.clear()
        self._populating = True
        try:
            with QSignalBlocker(self.tree):
                self.tree.clear()
                for row in self._rows:
                    self.tree.addTopLevelItem(self._item(row))
                for position, row in enumerate(self._rows):
                    item = self.tree.topLevelItem(position)
                    item.setSelected(row.index in selected)
                    if row.index == current:
                        self.tree.setCurrentItem(item, 0, QItemSelectionModel.NoUpdate)
        finally:
            self._populating = False
        self.tree.verticalScrollBar().setValue(scroll)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove if self._editable else QAbstractItemView.NoDragDrop)
        self.note_label.setText(note or ("" if self._rows else EMPTY_LIST))
        self.note_label.setVisible(bool(self.note_label.text()))
        self._update_buttons()
        self._push_box()

    def clear_document(self, note: str = NO_DOCUMENT) -> None:
        self._ticked.clear()
        self._hidden.clear()
        self.show_rows([], False, note)

    def _item(self, row: ObjectiveRow) -> QTreeWidgetItem:
        text, note = resolved_text(row)
        item = QTreeWidgetItem([str(row.order), "", "", "", "", "", " ".join(text.splitlines()), f"#{row.trigger_id} {row.name}"])
        item.setData(COL_ORDER, _INDEX_ROLE, row.index)
        item.setData(COL_ORDER, _RAW_ROLE, row.order)
        item.setData(COL_SHORT, _RAW_ROLE, row.short)
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        if self._editable:
            flags |= Qt.ItemIsEditable | Qt.ItemIsDragEnabled
        item.setFlags(flags)
        for column, _label, attr, _tip in FLAG_COLUMNS:
            value = {"header": row.header, "display_as_objective": row.objective,
                     "display_on_screen": row.on_screen, "mute_objectives": row.mute}[attr]
            item.setCheckState(column, Qt.Checked if value else Qt.Unchecked)
        shown = row.index not in self._hidden
        item.setCheckState(COL_PREVIEW, Qt.Checked if shown else Qt.Unchecked)
        item.setToolTip(COL_PREVIEW, PREVIEW_TIP if shown else HIDDEN_TIP)
        font = item.font(COL_SHORT)
        font.setBold(row.header)
        font.setItalic(bool(note) and text != row.short)
        item.setFont(COL_SHORT, font)
        if note:
            item.setToolTip(COL_SHORT, note + ("" if text == row.short else f"\nStored text: {row.short or '(empty)'}"))
        elif "\n" in row.short:
            item.setToolTip(COL_SHORT, "Several lines: edit it in Trigger Details")
        if not row.enabled:
            item.setForeground(COL_TRIGGER, QColor(140, 140, 140))
            item.setToolTip(COL_TRIGGER, DISABLED_TIP)
        if row.index in self._hidden:
            item.setForeground(COL_SHORT, QColor(140, 140, 140))
            item.setToolTip(COL_SHORT, HIDDEN_TIP)
        return item

    def rows(self) -> list[ObjectiveRow]:
        return list(self._rows)

    def editable(self) -> bool:
        return self._editable

    # -- selection ------------------------------------------------------------

    def selected_indices(self) -> list[int]:
        """Selected rows' trigger indices, in list order."""
        return [
            self.tree.topLevelItem(position).data(COL_ORDER, _INDEX_ROLE)
            for position in range(self.tree.topLevelItemCount())
            if self.tree.topLevelItem(position).isSelected()
        ]

    def current_index(self) -> int | None:
        item = self.tree.currentItem()
        return None if item is None else item.data(COL_ORDER, _INDEX_ROLE)

    def select_index(self, index: int | None) -> None:
        """Make the row for trigger `index` the selection, or clear it, without reporting back."""
        if index in self.selected_indices() and index == self.current_index():
            return
        with QSignalBlocker(self.tree):
            self.tree.clearSelection()
            for position in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(position)
                if item.data(COL_ORDER, _INDEX_ROLE) == index:
                    self.tree.setCurrentItem(item)
                    item.setSelected(True)
                    self.tree.scrollToItem(item)
                    break
            else:
                self.tree.setCurrentItem(None)
        self._update_buttons()

    def select_indices(self, indices: Sequence[int]) -> None:
        wanted = set(indices)
        with QSignalBlocker(self.tree):
            self.tree.clearSelection()
            for position in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(position)
                if item.data(COL_ORDER, _INDEX_ROLE) in wanted:
                    item.setSelected(True)
        self._update_buttons()

    def _on_current_changed(self, current, _previous) -> None:
        if not self._populating and current is not None:
            self.on_select(current.data(COL_ORDER, _INDEX_ROLE))

    # -- edits ----------------------------------------------------------------

    def later(self, callback: Callable, *args) -> None:
        """Run `callback(*args)` once control is back in the event loop."""
        self._pending.append((callback, args))
        self._flush_timer.start()

    def flush(self) -> None:
        pending, self._pending = self._pending, []
        for callback, args in pending:
            callback(*args)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._populating or (column not in _FLAG_ATTR and column != COL_PREVIEW):
            return
        index = item.data(COL_ORDER, _INDEX_ROLE)
        value = item.checkState(column) == Qt.Checked
        selected = self.selected_indices()
        indices = selected if index in selected else [index]
        if column == COL_PREVIEW:
            self.later(self.set_hidden, indices, not value)
        else:
            self.later(self.on_flags, indices, _FLAG_ATTR[column], value)

    def drop_rows(self, indices: Sequence[int], position: int) -> None:
        """A drop of the rows for `indices` at list `position` (counted with them in place)."""
        if not self._editable or not indices:
            return
        wanted = set(indices)
        rest_position = sum(1 for row in self._rows[:position] if row.index not in wanted)
        changes = orders_after_move(self._rows, wanted, rest_position)
        if changes:
            self.later(self.on_order, changes, "Reorder objectives")

    def move_selected(self, delta: int) -> None:
        changes = moved_by_step(self._rows, self.selected_indices(), delta)
        if changes:
            self.on_order(changes, "Reorder objectives")

    def _add_trigger(self) -> None:
        picked = self.pick_triggers(self.candidates())
        if picked:
            self.on_promote(picked)

    def pick_triggers(self, candidates: Sequence[tuple[int, str]]) -> list[int]:
        """The picker's choice, or [] on Cancel. Tests replace this."""
        dialog = _PickTriggers(self, candidates)
        try:
            return dialog.picked() if dialog.exec_() == QDialog.Accepted else []
        finally:
            dialog.deleteLater()

    def _update_buttons(self) -> None:
        editable = self._editable
        selected = self.selected_indices()
        positions = [position for position, row in enumerate(self._rows) if row.index in set(selected)]
        for button in (self.new_objective_button, self.new_header_button, self.add_trigger_button):
            button.setEnabled(editable)
        self.remove_button.setEnabled(editable and bool(selected))
        self.up_button.setEnabled(editable and bool(positions) and positions[0] > 0)
        self.down_button.setEnabled(editable and bool(positions) and positions[-1] < len(self._rows) - 1)
        self.renumber_button.setEnabled(editable and len(self._rows) > 0)

    # -- preview --------------------------------------------------------------

    def attach_overlay(self, overlay: ObjectivesOverlay) -> None:
        """The map's copy of the box: fed the same lines and ticks."""
        self._overlay = overlay
        overlay.state.on_tick = self.toggle_tick
        self._push_box()

    def preview_place(self) -> str:
        return self._place

    def set_preview_place(self, place: str) -> None:
        """Programmatic: shows the place without reporting it."""
        position = next((n for n, (value, _label) in enumerate(PLACES) if value == place), 0)
        with QSignalBlocker(self.place_combo):
            self.place_combo.setCurrentIndex(position)
        self._apply_place(PLACES[position][0])

    def _on_place_changed(self, _position: int) -> None:
        self._apply_place(self.place_combo.currentData())
        self.on_preview_place(self._place)

    def _apply_place(self, place: str) -> None:
        self._place = place
        self.preview.setVisible(place == PLACE_DOCK)

    def _on_section_moved(self, *_args) -> None:
        self._columns_moved = True
        self._columns_timer.start()

    def _on_section_resized(self, logical: int, *_args) -> None:
        self._columns_resized.add(logical)
        self._columns_timer.start()

    def _save_columns(self) -> None:
        """Saved when a column moved or one other than the stretched last one
        was resized: that one follows the tree's width, which is no user choice."""
        header = self.tree.header()
        last = header.logicalIndex(header.count() - 1)
        user_choice = self._columns_moved or bool(self._columns_resized - {last})
        self._columns_moved = False
        self._columns_resized.clear()
        if not user_choice:
            return
        state = bytes(header.saveState())
        if state != self._saved_columns:
            self._saved_columns = state
            self.on_columns(state)

    def shutdown(self) -> None:
        """The window closes: a pending column save must not land in a later config."""
        self._columns_timer.stop()
        self._flush_timer.stop()

    def refresh_icons(self) -> None:
        """A theme change: the eye column's header icon in the new colours."""
        self.tree.headerItem().setIcon(COL_PREVIEW, ui_icons.icon(PREVIEW_ICON))

    def restore_columns(self, state: bytes | None) -> None:
        """A saved column order and widths; one that will not restore, or names
        another column count, leaves the default."""
        header = self.tree.header()
        default = header.saveState()
        if not state:
            return
        with QSignalBlocker(header):
            if not header.restoreState(state) or header.count() != COL_TRIGGER + 1:
                header.restoreState(default)
        self._saved_columns = bytes(header.saveState())

    def set_variables(self, names: Sequence[str]) -> None:
        """The document's named trigger variables, which a <name> tag can show."""
        self._variables = tuple(name for name in names if name)
        self._push_box()

    def variable_mode(self) -> str:
        return self._variable_mode

    def set_variable_mode(self, mode: str) -> None:
        """Programmatic: shows the mode without reporting it."""
        position = next((n for n, (value, _label) in enumerate(VARIABLE_MODES) if value == mode), 0)
        with QSignalBlocker(self.variable_combo):
            self.variable_combo.setCurrentIndex(position)
        self._variable_mode = VARIABLE_MODES[position][0]
        self._push_box()

    def _on_variable_mode_changed(self, _position: int) -> None:
        self._variable_mode = self.variable_combo.currentData()
        self._push_box()
        self.on_variable_mode(self._variable_mode)

    def at_start(self) -> bool:
        return self._at_start

    def set_at_start(self, at_start: bool) -> None:
        with QSignalBlocker(self.start_radio), QSignalBlocker(self.all_radio):
            (self.start_radio if at_start else self.all_radio).setChecked(True)
        self._at_start = bool(at_start)
        self._push_box()

    def _on_at_start_toggled(self, checked: bool) -> None:
        self._at_start = bool(checked)
        self._push_box()
        self.on_at_start(self._at_start)

    def ticked(self) -> set[int]:
        return set(self._ticked)

    def toggle_tick(self, index: int) -> None:
        self._ticked ^= {index}
        self._push_box()

    def hidden(self) -> set[int]:
        return set(self._hidden)

    def set_hidden(self, indices: Collection[int], hidden: bool) -> None:
        """Leave rows out of the preview, or bring them back. The list and the scenario are untouched."""
        if hidden:
            self._hidden |= set(indices)
        else:
            self._hidden -= set(indices)
        self.show_rows(self._rows, self._editable, self.note_label.text() if self._rows else "")

    def preview_typing(self, indices: Sequence[int], text: str) -> None:
        """Trigger Details' short description as it is being typed: the box and
        the rows' cells follow it; the next repopulate (its commit) drops it."""
        rows = {row.index: row for row in self._rows}
        changed = False
        for index in indices:
            row = rows.get(index)
            if row is None or self._typing.get(index, row.short) == text:
                continue
            self._typing[index] = text
            changed = True
            shown, _note = resolved_text(row._replace(short=text))
            for position in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(position)
                if item.data(COL_ORDER, _INDEX_ROLE) == index:
                    with QSignalBlocker(self.tree):
                        item.setText(COL_SHORT, " ".join(shown.splitlines()))
        if changed:
            self._push_box()

    def clear_preview(self) -> None:
        """Another document or a structural edit: ticks and hidden rows go."""
        self._ticked.clear()
        self._hidden.clear()
        self.show_rows(self._rows, self._editable, self.note_label.text() if self._rows else "")

    def _show_context_menu(self, pos) -> None:
        selected = self.selected_indices()
        menu = QMenu(self.tree)
        hide = menu.addAction("Hide in Preview")
        hide.setEnabled(any(index not in self._hidden for index in selected))
        hide.triggered.connect(lambda: self.set_hidden(selected, True))
        show = menu.addAction("Show in Preview")
        show.setEnabled(any(index in self._hidden for index in selected))
        show.triggered.connect(lambda: self.set_hidden(selected, False))
        show_all = menu.addAction("Show All in Preview")
        show_all.setEnabled(bool(self._hidden))
        show_all.triggered.connect(lambda: self.set_hidden(list(self._hidden), False))
        menu.addSeparator()
        with_ids = [row.index for row in self._rows if row.index in set(selected) and has_string_id(row.short_stid)]
        use_text = menu.addAction("Use My Text (Clear String ID)")
        use_text.setToolTip("The game shows these rows' language strings, not their text; clear the ids so the text shows")
        use_text.setEnabled(self._editable and bool(with_ids))
        use_text.triggered.connect(lambda: self.on_clear_string_ids(with_ids))
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def _push_box(self) -> None:
        entries = box_entries(self._rows, self._at_start, self._hidden, self._typing)
        self.box.set_entries(entries, self._ticked, self._variables, self._variable_mode)
        if self._overlay is not None:
            self._overlay.set_entries(entries, self._ticked, self._variables, self._variable_mode)
