"""Widgets for picking an id-dataset catalog entry (descape/object_catalog.py)
by name instead of typing a raw integer: CatalogLineEdit (the fast path, type-
ahead) and CatalogBrowseDialog (a category-grouped tree, opened from the
line edit's "..." button).

Both dumb and callback-driven like TriggerPanel itself (see trigger_fields.py's
own module docstring): neither touches EditHistory or writes anything back on
its own. CatalogLineEdit reports a committed value through its `committed`
signal, and TriggerPanel decides what that means.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
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

from descape import object_catalog, unit_sprites

# Neutral preview: GAIA's own team_index (unit_sprites.py's own team_index
# docstring), so a preview never implies a player that picking this entry has
# nothing to do with. Facing the same direction every time, for the same
# reason -- there is no "this player's" rotation to prefer here either.
_PREVIEW_TEAM_INDEX = 0
_PREVIEW_ROTATION = 0.0
_PREVIEW_HALF_W = 32
_PREVIEW_BOX_PX = 2 * _PREVIEW_HALF_W
# Plain white, since this preview is never composited over the map: AGENTS.md's
# own hard rule on composite buildings (a standing_graphic can be one small
# annex, not the whole thing -- confirmed on Town Center, whose own entry is
# just its "back" piece) is why this uses sprite_pieces_for() and pastes every
# piece, not sprite_for()'s single graphic.
_PREVIEW_BG = np.array([255, 255, 255], dtype=np.uint8)

# Past Qt.UserRole (which already holds the object id on column 0), for
# CatalogBrowseDialog's "Show editor-hidden objects" filter.
_HIDDEN_ROLE = Qt.UserRole + 1


def _composite_pieces(pieces: list[unit_sprites.SpritePiece]) -> np.ndarray | None:
    """Every piece alpha-blended onto one opaque canvas sized to fit them
    all, or None for an empty list. Same integer "over" blend render.py's
    own _clipped_paint_rgba uses, without the clip: this canvas is always
    sized to exactly contain every piece, so there is nothing to clip."""
    if not pieces:
        return None
    boxes = []
    for piece in pieces:
        height, width = piece.draw.rgba.shape[:2]
        x0 = piece.dx - piece.draw.hotspot_x
        y0 = piece.dy - piece.draw.hotspot_y
        boxes.append((x0, y0, x0 + width, y0 + height))
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)

    canvas = np.empty((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)
    canvas[:, :] = _PREVIEW_BG
    for piece, (x0, y0, _x1, _y1) in zip(pieces, boxes):
        oy, ox = y0 - min_y, x0 - min_x
        rgba = piece.draw.rgba
        height, width = rgba.shape[:2]
        alpha = rgba[..., 3:4].astype(np.uint16)
        dst = canvas[oy : oy + height, ox : ox + width]
        dst[...] = (
            (rgba[..., :3].astype(np.uint16) * alpha + dst.astype(np.uint16) * (255 - alpha)) // 255
        ).astype(np.uint8)
    return canvas


def _preview_pixmap(object_id: int) -> QPixmap | None:
    """object_id's sprite as a QPixmap scaled to fit the preview box, or None
    if unit_sprites has nothing for it -- no install configured, no graphic
    mapped (every tech, some objects), or a decode failure.
    sprite_pieces_for() already turns every one of those into an empty list
    rather than raising."""
    pieces = unit_sprites.sprite_pieces_for(
        object_id, _PREVIEW_ROTATION, _PREVIEW_TEAM_INDEX, _PREVIEW_HALF_W
    )
    canvas = _composite_pieces(pieces)
    if canvas is None:
        return None
    canvas = np.ascontiguousarray(canvas)
    height, width = canvas.shape[:2]
    image = QImage(canvas.data, width, height, 3 * width, QImage.Format_RGB888)
    # QPixmap.fromImage() copies the pixel data, so `canvas` going out of
    # scope afterward (it is a local) can't leave the pixmap holding a
    # dangling buffer.
    pixmap = QPixmap.fromImage(image)
    return pixmap.scaled(
        _PREVIEW_BOX_PX, _PREVIEW_BOX_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


class CatalogBrowseDialog(QDialog):
    """A category-grouped tree over one id-dataset catalog, filtered by name.

    Reuses TriggerPanel's own vocabulary-picker shape (_build_picker_page(),
    _apply_picker_filter()): single column at ResizeToContents, groups
    expanded, double-click to accept, "hide a group left with nothing under
    it" filtering. A dialog rather than another detail_stack page: this sets
    one field on a form the user is already looking at, where the vocabulary
    picker replaces the whole entry list while adding a new condition/effect.

    "Show editor-hidden objects" starts unchecked, so `hide_in_editor`
    entries stay out of the tree until opted into.
    """

    def __init__(
        self,
        catalog: Sequence[object_catalog.CatalogEntry],
        default_category: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Browse")
        self.resize(420, 480)
        self._catalog = catalog
        self._selected_id: int | None = None

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)

        self.show_hidden_checkbox = QCheckBox("Show editor-hidden objects")
        self.show_hidden_checkbox.toggled.connect(lambda *_: self._apply_filter(self.filter_edit.text()))

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name"])
        self.tree.setColumnCount(1)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.itemDoubleClicked.connect(lambda *_: self._accept_current())
        self.tree.currentItemChanged.connect(lambda *_: self._update_preview())

        self.preview = QLabel()
        self.preview.setFixedSize(_PREVIEW_BOX_PX, _PREVIEW_BOX_PX)
        self.preview.setAlignment(Qt.AlignCenter)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_current)
        buttons.rejected.connect(self.reject)

        preview_row = QHBoxLayout()
        preview_row.addWidget(self.tree, stretch=1)
        preview_row.addWidget(self.preview)

        layout = QVBoxLayout(self)
        layout.addWidget(self.filter_edit)
        layout.addWidget(self.show_hidden_checkbox)
        layout.addLayout(preview_row)
        layout.addWidget(buttons)

        self._populate(default_category)
        self._apply_filter(self.filter_edit.text())

    def _populate(self, default_category: str) -> None:
        self.tree.clear()
        by_category: dict[str, list[object_catalog.CatalogEntry]] = {}
        for candidate in self._catalog:
            by_category.setdefault(candidate.category, []).append(candidate)
        for category, entries in by_category.items():
            group = QTreeWidgetItem([f"{category} ({len(entries)})"])
            self.tree.addTopLevelItem(group)
            for candidate in sorted(entries, key=lambda e: e.name):
                child = QTreeWidgetItem([candidate.name])
                child.setData(0, Qt.UserRole, candidate.id)
                child.setData(0, _HIDDEN_ROLE, candidate.hidden)
                group.addChild(child)
            group.setExpanded(not default_category or category == default_category)

    def _apply_filter(self, text: str) -> None:
        """Hide non-matching rows, editor-hidden rows unless opted into, and
        any group left with nothing under it."""
        needle = text.strip().lower()
        show_hidden = self.show_hidden_checkbox.isChecked()
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            shown = 0
            for c in range(group.childCount()):
                child = group.child(c)
                text_mismatch = bool(needle) and needle not in child.text(0).lower()
                editor_hidden = bool(child.data(0, _HIDDEN_ROLE)) and not show_hidden
                is_hidden = text_mismatch or editor_hidden
                child.setHidden(is_hidden)
                shown += not is_hidden
            group.setHidden(shown == 0)
            if needle and shown:
                group.setExpanded(True)

    def _update_preview(self) -> None:
        item = self.tree.currentItem()
        object_id = item.data(0, Qt.UserRole) if item is not None else None
        # Object ids and tech ids are separate id spaces that can collide on
        # the same number -- sprite_pieces_for() would happily return an
        # unrelated unit's graphic for a tech id that coincidentally matches
        # one. Only ever preview a real object-catalog entry.
        found = object_catalog.entry(self._catalog, object_id) if isinstance(object_id, int) else None
        pixmap = _preview_pixmap(object_id) if found is not None and found.category != "Techs" else None
        self.preview.setPixmap(pixmap if pixmap is not None else QPixmap())

    def select(self, object_id: int) -> None:
        """Selects and reveals the row for object_id, if this catalog has one."""
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            for c in range(group.childCount()):
                child = group.child(c)
                if child.data(0, Qt.UserRole) == object_id:
                    group.setExpanded(True)
                    self.tree.setCurrentItem(child)
                    return

    def selected_id(self) -> int | None:
        return self._selected_id

    def _accept_current(self) -> None:
        item = self.tree.currentItem()
        if item is None or item.isHidden():
            return
        data = item.data(0, Qt.UserRole)
        if data is None:  # a group heading
            return
        self._selected_id = data
        self.accept()


class CatalogLineEdit(QWidget):
    """A QLineEdit + type-ahead completer + "..." browse button, for picking
    one id-dataset catalog entry.

    Accepts a typed name (case-insensitively, matched against the catalog) or
    a raw integer, so an id the catalog does not cover -- always possible,
    since it is library-backed until slice 4 -- is still settable. Reports a
    new value through `committed` only on editingFinished or completer
    activation, matching the plain QLineEdit field's own commit-on-focus-out
    behaviour rather than firing on every keystroke.
    """

    committed = pyqtSignal(int)

    def __init__(
        self,
        catalog: Sequence[object_catalog.CatalogEntry],
        default_category: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._catalog = catalog
        self._default_category = default_category
        self._value: int | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.line_edit = QLineEdit()
        # Grayed-out hint text on an empty field, gone the moment there is a
        # real value -- the QLineEdit-native equivalent of _make_spinbox's
        # special_value_text, for a caller that reports "no value" as None.
        self.line_edit.setPlaceholderText("(unset)")
        completer = QCompleter([entry.name for entry in catalog], self.line_edit)
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
        self.browse_button.setFixedWidth(28)
        self.browse_button.setToolTip("Browse…")
        self.browse_button.clicked.connect(self._browse)

        layout.addWidget(self.line_edit, stretch=1)
        layout.addWidget(self.browse_button)

    def value(self) -> int | None:
        return self._value

    def set_value(self, object_id: int | None) -> None:
        """Sets the displayed value without emitting `committed` -- the
        populate-time path. Callers connect to `committed` after calling this,
        matching every other widget in TriggerPanel._build_widget()."""
        self._value = object_id
        if object_id is None:
            self.line_edit.setText("")
            return
        name = object_catalog.name_for(self._catalog, object_id)
        self.line_edit.setText(name or str(object_id))

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 (Qt override)
        super().setEnabled(enabled)
        self.line_edit.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)

    def _commit_from_text(self, text: str | None = None) -> None:
        typed = (text if text is not None else self.line_edit.text()).strip()
        if not typed:
            return
        if typed.lstrip("-").isdigit():
            new_value = int(typed)
        else:
            match = next(
                (e for e in self._catalog if e.name.lower() == typed.lower()), None
            )
            if match is None:
                # No such name and not a raw integer either -- reject the
                # edit rather than write a value the user never chose, and
                # restore the field to what it actually holds.
                self.set_value(self._value)
                return
            new_value = match.id
        self._commit(new_value)

    def _browse(self) -> None:
        dialog = CatalogBrowseDialog(self._catalog, self._default_category, self)
        if self._value is not None:
            dialog.select(self._value)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_id() is not None:
            self._commit(dialog.selected_id())

    def _commit(self, new_value: int) -> None:
        if new_value == self._value:
            self.set_value(self._value)  # revert any half-typed text
            return
        self.set_value(new_value)
        self.committed.emit(new_value)
