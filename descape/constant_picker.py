"""Widgets for picking an id-dataset catalog entry (descape/object_catalog.py)
by name instead of typing a raw integer: CatalogLineEdit (the fast path, type-
ahead) and CatalogBrowseDialog (the Units catalog as a modal, GH #273, opened
from an object field's "..." button).

CatalogLineEdit is a thin adapter over descape/value_picker.py's domain-free
ValueLineEdit. CatalogBrowseDialog hosts catalog_browser.CatalogBrowserView
and a unit_card.UnitCard. A tech field's "..." stays value_picker's plain
tree, since a tech id is not a const: a portrait or stats would answer for an
unrelated object. This module also keeps the one piece of domain knowledge
value_picker.py has none of, a sprite preview, which the Disables dialog's
Full List, the Cliff row and the facet preview use. Dumb and callback-driven
like TriggerPanel itself (see trigger_fields.py's own module docstring):
neither touches EditHistory or writes anything back on its own.
CatalogLineEdit reports a committed value through its `committed` signal, and
TriggerPanel decides what that means.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

from descape import catalog_thumbs, object_catalog, settings, unit_sprites
from descape.catalog_browser import HIDDEN_LABEL, CatalogBrowserView
from descape.unit_card import UnitCard
from descape.value_picker import PickerItem, ValueLineEdit

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

# The trigger tech picker's box: object_catalog.INTERNAL_TECH_FAMILIES' rows.
INTERNAL_TECHS_LABEL = "Show internal techs"
# Ten icon rows above the card.
_BROWSE_SIZE = (460, 640)


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
    for piece, (x0, y0, _x1, _y1) in zip(pieces, boxes, strict=True):
        oy, ox = y0 - min_y, x0 - min_x
        rgba = piece.draw.rgba
        height, width = rgba.shape[:2]
        alpha = rgba[..., 3:4].astype(np.uint16)
        dst = canvas[oy : oy + height, ox : ox + width]
        dst[...] = (
            (rgba[..., :3].astype(np.uint16) * alpha + dst.astype(np.uint16) * (255 - alpha)) // 255
        ).astype(np.uint8)
    return canvas


def preview_pixmap(object_id: int, rotation: float = _PREVIEW_ROTATION) -> QPixmap | None:
    """object_id's sprite as a QPixmap scaled to fit the preview box, or None
    if unit_sprites has nothing for it -- no install configured, no graphic
    mapped (every tech, some objects), or a decode failure.
    sprite_pieces_for() already turns every one of those into an empty list
    rather than raising.

    `rotation` defaults to the facing-neutral 0.0 every existing caller here
    wants; the Cliff tool's own param-row preview (viewer.py) passes the
    user's chosen frame instead, since rotation picks a cliff's SHAPE, not a
    facing -- see unit_sprites.rotation_is_variant()'s own docstring.
    """
    pieces = unit_sprites.sprite_pieces_for(
        object_id, rotation, _PREVIEW_TEAM_INDEX, _PREVIEW_HALF_W
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


def items_for(catalog: Sequence[object_catalog.CatalogEntry], hide_internal: bool = False) -> tuple[PickerItem, ...]:
    """catalog's own CatalogEntry rows reduced to value_picker.py's
    domain-free PickerItem shape. `class_id` is dropped -- no widget here
    ever read it. `hide_internal` folds `internal` into `hidden`."""
    return tuple(
        PickerItem(
            label=entry.name,
            value=entry.id,
            group=entry.category,
            hidden=entry.hidden or (hide_internal and entry.internal),
            search_text=entry.search_text,
        )
        for entry in catalog
    )


def catalog_preview(item: PickerItem) -> QPixmap | None:
    """A PickerItem's sprite preview, or None for a Techs entry.

    Object ids and tech ids are separate id spaces that can collide on the
    same number -- sprite_pieces_for() would happily return an unrelated
    unit's graphic for a tech id that coincidentally matches one.
    `item.group` already carries "Techs" for every tech-catalog row
    (object_catalog.techs()), so this never needs to look the id back up in
    a catalog to know which space it came from.
    """
    if item.group == "Techs":
        return None
    return preview_pixmap(item.value)


class CatalogBrowseDialog(QDialog):
    """The Units catalog over one object catalog (GH #273): category buttons,
    class combo, filter and three views, with the Placing card under the list.

    "Show editor-hidden objects" starts unchecked, so `hide_in_editor` entries
    stay out of the list until opted into, select()'s own value excepted.

    OK returns what the card shows. A category switch drops the list's current
    row with signals blocked, so the pick is held here, as UnitsPanel holds its
    pending object. Enter in the filter is the exception: it could accept a row
    the filter just hid, so it moves to the first match and never accepts.
    """

    def __init__(
        self,
        catalog: Sequence[object_catalog.CatalogEntry],
        default_category: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Browse")
        self.resize(*_BROWSE_SIZE)
        self._catalog = catalog
        self._picked: int | None = None
        self._selected_id: int | None = None

        loader = catalog_thumbs.ThumbLoader(self)
        self.view = CatalogBrowserView(
            loader,
            self,
            entries=catalog,
            default_category=default_category,
            mode_store=(settings.get_browse_catalog_mode, settings.set_browse_catalog_mode),
        )
        self.view.current_changed.connect(self._on_current_changed)
        self.view.activated.connect(self._on_activated)
        self.filter_edit = self.view.filter_edit
        self.show_hidden_checkbox = self.view.show_hidden_checkbox
        self.card = UnitCard(stable=True, loader=loader)
        self.card.clear("No object picked")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_current)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.card)
        layout.addWidget(buttons)

        # Here, not in the view: the sidebar's policies keep the map's arrow keys.
        for button in (*self.view.category_buttons.values(), *self.view.display_buttons.values()):
            button.setFocusPolicy(Qt.TabFocus)
        self.view.class_combo.setFocusPolicy(Qt.StrongFocus)
        self.view.list.setFocusPolicy(Qt.StrongFocus)
        # The tab order now starts on a button.
        self.filter_edit.setFocus()
        # Each open holds a populated list and a memo, and callers parent it to long-lived widgets.
        self.finished.connect(self.deleteLater)

    def select(self, value: int) -> None:
        """Opens on `value`, the field's current one: pinned, so a hidden row stays pickable."""
        self.view.pin(value)
        self.view.select(value)

    def selected_id(self) -> int | None:
        return self._selected_id

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.focusWidget() is self.filter_edit:
            self.view.focus_first_row()
            return
        super().keyPressEvent(event)

    def _on_current_changed(self, value: int | None) -> None:
        if value is None:
            return
        self._picked = value
        # The row's own name: display_name() can spell it differently.
        self.card.show_const(value, name=object_catalog.name_for(self._catalog, value))

    def _on_activated(self, value: int) -> None:
        self._picked = value
        self._accept_current()

    def _accept_current(self) -> None:
        if self._picked is None:
            return
        self._selected_id = self._picked
        self.accept()


class CatalogLineEdit(ValueLineEdit):
    """A QLineEdit + type-ahead completer + "..." browse button, for picking
    one id-dataset catalog entry.

    Accepts a typed name (case-insensitively, matched against the catalog) or
    a raw integer, so an id the catalog does not cover -- always possible,
    since it is library-backed until slice 4 -- is still settable.

    `hide_internal` (the trigger tech picker) keeps internal techs out of
    Browse until INTERNAL_TECHS_LABEL is ticked; a typed name still matches
    them, as it matches editor-hidden objects.

    Browse is CatalogBrowseDialog for an object catalog, narrowed by
    set_group(), and the base class's plain tree for a tech one.
    """

    def __init__(
        self,
        catalog: Sequence[object_catalog.CatalogEntry],
        default_category: str = "",
        parent=None,
        placeholder: str = "(unset)",
        hide_internal: bool = False,
    ):
        self._catalog = catalog
        self._default_category = default_category
        super().__init__(
            items_for(catalog, hide_internal),
            default_group=default_category,
            show_values=False,
            hidden_label=INTERNAL_TECHS_LABEL if hide_internal else HIDDEN_LABEL,
            browse_title="Browse",
            # A parameter so a multi-entry form can say "(differs)" (GH #60).
            placeholder=placeholder,
            parent=parent,
        )

    def _browse(self) -> None:
        if any(row.category == "Techs" for row in self._catalog):
            super()._browse()
            return
        rows = [row for row in self._catalog if row.category == self._group] if self._group else self._catalog
        dialog = CatalogBrowseDialog(rows, self._group or self._default_category, parent=self)
        if self._value is not None:
            dialog.select(self._value)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_id() is not None:
            self.commit_value(dialog.selected_id())
