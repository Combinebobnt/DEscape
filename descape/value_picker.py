"""Generic (label, value) picker, domain-free: PickerItem, the filterable
tree view ValuePickerView, its modal shell ValueBrowseDialog, and the
type-ahead ValueLineEdit built on top.

Reproduces two existing widgets' exact behaviour with their domain coupling
lifted out. descape/constant_picker.py's CatalogBrowseDialog/CatalogLineEdit
subclass these for the object-catalog case (a sprite preview, an
editor-hidden filter, ids rendered bare); a converted large-ENUM field in
descape/trigger_panel.py and the Units-mode sidebar catalog are two more
consumers, neither needing the same trimmings. Dumb like every other widget
in this family: no EditHistory, no model, just `committed`/`activated`
signals the caller decides what to do with.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QCompleter,
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
    QWidget,
)

# Past Qt.UserRole (which already holds the row's value on column 0).
_HIDDEN_ROLE = Qt.UserRole + 1
_SEARCH_ROLE = Qt.UserRole + 2

_PREVIEW_BOX_PX = 64


@dataclass(frozen=True)
class PickerItem:
    """One pickable row. `group=""` on every item means a flat tree with no
    group headings; `hidden=True` is filtered out unless a "show hidden" box
    (when the caller asks for one via `hidden_label`) is ticked."""

    label: str
    value: int
    group: str = ""
    hidden: bool = False
    search_text: str = ""

    def __post_init__(self) -> None:
        if not self.search_text:
            object.__setattr__(self, "search_text", f"{self.label.lower()} {self.value}")


def _rendered_text(item: PickerItem, show_values: bool) -> str:
    """`show_values=False` (the object-catalog rendering): the bare label.
    `show_values=True` (a converted enum): "label (value)", matching what
    the QComboBox it replaces already rendered."""
    return f"{item.label} ({item.value})" if show_values else item.label


class _HScrollStableTreeWidget(QTreeWidget):
    """QTreeWidget that ignores Qt's horizontal auto-scroll on selection.

    Qt's default scrollTo(index, EnsureVisible) chases a ResizeToContents
    column's full width, so selecting a long row yanks the horizontal
    scrollbar to that row's far-right edge. Restoring the horizontal
    position after the base implementation runs keeps its vertical
    auto-scroll (needed for keyboard navigation) while dropping the
    horizontal jump, regardless of what triggered the scroll.
    """

    def scrollTo(self, index, hint=QTreeWidget.EnsureVisible) -> None:
        hbar = self.horizontalScrollBar()
        pos = hbar.value()
        super().scrollTo(index, hint)
        hbar.setValue(pos)


def _configure_scrolling(tree: QTreeWidget) -> None:
    """Make a tree scroll horizontally instead of eliding.

    All three calls are load-bearing together. With stretchLastSection on,
    a QTreeWidget never scrolls horizontally: it squeezes the columns and
    elides instead, which comes back as "there is a scrollbar but the text
    is still cut" if only the policy is set.
    """
    tree.header().setStretchLastSection(False)
    tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    tree.setTextElideMode(Qt.ElideNone)


class ValuePickerView(QWidget):
    """Filter box + optional show-hidden box + grouped-or-flat tree +
    optional preview. No buttons, no dialog, no commit semantics -- both
    ValueBrowseDialog (adds Ok/Cancel) and a standalone always-visible
    sidebar page are built on top of this.
    """

    activated = pyqtSignal(int)  # itemActivated: double-click AND Enter
    current_changed = pyqtSignal(object)  # int | None

    def __init__(
        self,
        items: Sequence[PickerItem],
        *,
        default_group: str = "",
        show_values: bool = False,
        hidden_label: str = "",
        preview: Callable[[PickerItem], QPixmap | None] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._show_values = show_values
        self._preview_hook = preview
        self._items: tuple[PickerItem, ...] = ()
        self._by_value: dict[int, PickerItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter_edit)

        # Omitted entirely rather than always present-but-disabled: a
        # converted enum has no hidden members at all, so there is nothing
        # for the box to do.
        self.show_hidden_checkbox: QCheckBox | None = None
        if hidden_label:
            self.show_hidden_checkbox = QCheckBox(hidden_label)
            self.show_hidden_checkbox.toggled.connect(
                lambda *_: self._apply_filter(self.filter_edit.text())
            )
            layout.addWidget(self.show_hidden_checkbox)

        self.tree = _HScrollStableTreeWidget()
        self.tree.setHeaderLabels(["Name"])
        self.tree.setColumnCount(1)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        _configure_scrolling(self.tree)
        # itemActivated, not itemDoubleClicked: it fires on Enter too, and a
        # sidebar host has no Ok button to fall back on.
        self.tree.itemActivated.connect(self._on_activated)
        self.tree.currentItemChanged.connect(lambda *_: self._on_current_changed())

        # Omitted (no box, no layout slot) rather than an empty QLabel: an
        # enum picker has no sprite to show and would otherwise carry a dead
        # 64 px column.
        self.preview: QLabel | None = None
        tree_row = QHBoxLayout()
        tree_row.addWidget(self.tree, stretch=1)
        if preview is not None:
            self.preview = QLabel()
            self.preview.setFixedSize(_PREVIEW_BOX_PX, _PREVIEW_BOX_PX)
            self.preview.setAlignment(Qt.AlignCenter)
            self.tree.currentItemChanged.connect(lambda *_: self._update_preview())
            tree_row.addWidget(self.preview)
        layout.addLayout(tree_row)

        self.set_items(items, default_group=default_group)

    # -- population -----------------------------------------------------

    def set_items(self, items: Sequence[PickerItem], *, default_group: str = "") -> None:
        self._items = tuple(items)
        self._by_value = {item.value: item for item in self._items}
        self.tree.clear()
        if self._grouped():
            self._populate_grouped(default_group)
        else:
            self._populate_flat()
        self._apply_filter(self.filter_edit.text())

    def _grouped(self) -> bool:
        return any(item.group for item in self._items)

    def _populate_grouped(self, default_group: str) -> None:
        by_group: dict[str, list[PickerItem]] = {}
        for item in self._items:
            by_group.setdefault(item.group, []).append(item)
        for group, members in by_group.items():
            heading = QTreeWidgetItem([f"{group} ({len(members)})"])
            self.tree.addTopLevelItem(heading)
            for item in sorted(members, key=lambda i: i.label):
                heading.addChild(self._make_row(item))
            heading.setExpanded(not default_group or group == default_group)

    def _populate_flat(self) -> None:
        # Caller order preserved, not re-sorted: a converted enum's
        # spec.choices order is meaningful (vocabulary-defined), and the
        # object catalog is already alphabetical at the source.
        for item in self._items:
            self.tree.addTopLevelItem(self._make_row(item))

    def _make_row(self, item: PickerItem) -> QTreeWidgetItem:
        row = QTreeWidgetItem([_rendered_text(item, self._show_values)])
        row.setData(0, Qt.UserRole, item.value)
        row.setData(0, _HIDDEN_ROLE, item.hidden)
        row.setData(0, _SEARCH_ROLE, item.search_text)
        return row

    # -- filtering --------------------------------------------------------

    def _apply_filter(self, text: str) -> None:
        """Hide non-matching rows, editor-hidden rows unless opted into, and
        (grouped mode only) any group left with nothing under it."""
        needle = text.strip().lower()
        show_hidden = self.show_hidden_checkbox is not None and self.show_hidden_checkbox.isChecked()
        if self._grouped():
            for i in range(self.tree.topLevelItemCount()):
                group = self.tree.topLevelItem(i)
                shown = 0
                for c in range(group.childCount()):
                    child = group.child(c)
                    is_hidden = self._row_hidden(child, needle, show_hidden)
                    child.setHidden(is_hidden)
                    shown += not is_hidden
                group.setHidden(shown == 0)
                if needle and shown:
                    group.setExpanded(True)
        else:
            for i in range(self.tree.topLevelItemCount()):
                row = self.tree.topLevelItem(i)
                row.setHidden(self._row_hidden(row, needle, show_hidden))

    @staticmethod
    def _row_hidden(row: QTreeWidgetItem, needle: str, show_hidden: bool) -> bool:
        text_mismatch = bool(needle) and needle not in (row.data(0, _SEARCH_ROLE) or "")
        editor_hidden = bool(row.data(0, _HIDDEN_ROLE)) and not show_hidden
        return text_mismatch or editor_hidden

    # -- selection ----------------------------------------------------------

    def select(self, value: int) -> None:
        """Selects and reveals the row for `value`, if there is one."""
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top.data(0, Qt.UserRole) == value:
                self.tree.setCurrentItem(top)
                return
            for c in range(top.childCount()):
                child = top.child(c)
                if child.data(0, Qt.UserRole) == value:
                    top.setExpanded(True)
                    self.tree.setCurrentItem(child)
                    return

    def current_value(self) -> int | None:
        """The current row's value, or None for no selection, a group
        heading (data(0, Qt.UserRole) is None -- 0 is a legal value and must
        not be treated as falsy here), or a row hidden by the current
        filter."""
        item = self.tree.currentItem()
        if item is None or item.isHidden():
            return None
        data = item.data(0, Qt.UserRole)
        return data if isinstance(data, int) else None

    def _on_current_changed(self) -> None:
        self.current_changed.emit(self.current_value())

    def _on_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        if item is None or item.isHidden():
            return
        value = item.data(0, Qt.UserRole)
        if value is None:  # a group heading
            return
        self.activated.emit(value)

    # -- preview --------------------------------------------------------

    def _update_preview(self) -> None:
        if self.preview is None:
            return
        item = self.tree.currentItem()
        data = item.data(0, Qt.UserRole) if item is not None else None
        found = self._by_value.get(data) if isinstance(data, int) else None
        pixmap = self._preview_hook(found) if found is not None else None
        self.preview.setPixmap(pixmap if pixmap is not None else QPixmap())


class ValueBrowseDialog(QDialog):
    """A thin modal shell around ValuePickerView: title, a fixed size, and
    an Ok/Cancel row. Aliases self.tree/self.filter_edit/
    self.show_hidden_checkbox/self.preview to the view's own widgets, as
    plain attributes rather than properties, so a caller (or a subclass'
    existing tests) poking those names keeps working unmodified.
    """

    def __init__(
        self,
        items: Sequence[PickerItem],
        *,
        default_group: str = "",
        show_values: bool = False,
        hidden_label: str = "",
        preview: Callable[[PickerItem], QPixmap | None] | None = None,
        title: str = "Select",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 480)
        self._selected_value: int | None = None

        self.view = ValuePickerView(
            items,
            default_group=default_group,
            show_values=show_values,
            hidden_label=hidden_label,
            preview=preview,
            parent=self,
        )
        self.view.activated.connect(lambda *_: self._accept_current())
        self.filter_edit = self.view.filter_edit
        self.show_hidden_checkbox = self.view.show_hidden_checkbox
        self.tree = self.view.tree
        self.preview = self.view.preview

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_current)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(buttons)

    def select(self, value: int) -> None:
        self.view.select(value)

    def selected_value(self) -> int | None:
        return self._selected_value

    def _accept_current(self) -> None:
        value = self.view.current_value()
        if value is None:
            return
        self._selected_value = value
        self.accept()


class ValueLineEdit(QWidget):
    """A QLineEdit + type-ahead completer + "..." browse button, for picking
    one (label, value) pair.

    Accepts a typed rendered or bare label (case-insensitively) or a raw
    integer, so a value the item list does not cover is still settable.
    Reports a new value through `committed` only on editingFinished or
    completer activation, matching the plain QLineEdit field's own
    commit-on-focus-out behaviour rather than firing on every keystroke.
    """

    committed = pyqtSignal(int)

    def __init__(
        self,
        items: Sequence[PickerItem],
        *,
        default_group: str = "",
        show_values: bool = False,
        hidden_label: str = "",
        preview: Callable[[PickerItem], QPixmap | None] | None = None,
        browse_title: str = "Select",
        placeholder: str = "(unset)",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._items = tuple(items)
        self._default_group = default_group
        self._show_values = show_values
        self._hidden_label = hidden_label
        self._preview = preview
        self._browse_title = browse_title
        self._value: int | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        completer = QCompleter(
            [_rendered_text(item, show_values) for item in self._items], self.line_edit
        )
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        self.line_edit.setCompleter(completer)
        self.line_edit.editingFinished.connect(self._commit_from_text)
        # activated(str) rather than activated() -- QCompleter emits the
        # picked completion's text, and editingFinished has not fired yet at
        # this point, so without this a mouse pick on the popup would need a
        # second focus-out before it committed.
        completer.activated[str].connect(self._commit_from_text)

        self.browse_button = QPushButton("…")
        # Not a bare 28: at Settings > Appearance's largest UI font the
        # ellipsis measures 27px in several families, which leaves it
        # touching both borders. Sized at construction, so a live font
        # change reaches it once the widget is next rebuilt.
        self.browse_button.setFixedWidth(max(28, self.browse_button.fontMetrics().horizontalAdvance("…") + 12))
        self.browse_button.setToolTip("Browse…")
        self.browse_button.clicked.connect(self._browse)

        layout.addWidget(self.line_edit, stretch=1)
        layout.addWidget(self.browse_button)

    def value(self) -> int | None:
        return self._value

    def set_value(self, value: int | None) -> None:
        """Sets the displayed value without emitting `committed` -- the
        populate-time path. Callers connect to `committed` after calling
        this, matching every other widget built this way."""
        self._value = value
        if value is None:
            self.line_edit.setText("")
            return
        item = self._by_value(value)
        if item is not None:
            self.line_edit.setText(_rendered_text(item, self._show_values))
        else:
            self.line_edit.setText(f"unknown ({value})" if self._show_values else str(value))
        # Show the start, not the end: setText() leaves the cursor past the
        # last character, so a label too long for the field read "nown (9999)".
        self.line_edit.setCursorPosition(0)

    # camelCase because it overrides QWidget.setEnabled, not a style slip.
    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.line_edit.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)

    def _by_value(self, value: int) -> PickerItem | None:
        return next((item for item in self._items if item.value == value), None)

    def _commit_from_text(self, text: str | None = None) -> None:
        typed = (text if text is not None else self.line_edit.text()).strip()
        if not typed:
            return
        if typed.lstrip("-").isdigit():
            # The raw-value escape hatch: settable even when the item list
            # (library-backed, or a fixed vocabulary) does not cover it.
            self._commit(int(typed))
            return
        needle = typed.lower()
        match = next(
            (
                item
                for item in self._items
                if _rendered_text(item, self._show_values).lower() == needle
                or item.label.lower() == needle
            ),
            None,
        )
        if match is None:
            # No such label and not a raw integer either -- reject the edit
            # rather than write a value the user never chose, and restore
            # the field to what it actually holds.
            self.set_value(self._value)
            return
        self._commit(match.value)

    def _browse(self) -> None:
        dialog = ValueBrowseDialog(
            self._items,
            default_group=self._default_group,
            show_values=self._show_values,
            hidden_label=self._hidden_label,
            preview=self._preview,
            title=self._browse_title,
            parent=self,
        )
        if self._value is not None:
            dialog.select(self._value)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_value() is not None:
            self._commit(dialog.selected_value())

    def _commit(self, new_value: int) -> None:
        if new_value == self._value:
            self.set_value(self._value)  # revert any half-typed text
            return
        self.set_value(new_value)
        self.committed.emit(new_value)
