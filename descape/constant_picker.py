"""Widgets for picking an id-dataset catalog entry (descape/object_catalog.py)
by name instead of typing a raw integer: CatalogLineEdit (the fast path, type-
ahead) and CatalogBrowseDialog (a category-grouped tree, opened from the
line edit's "..." button).

Both are thin adapters over descape/value_picker.py's domain-free
ValueLineEdit/ValueBrowseDialog: this module supplies the one piece of
domain knowledge value_picker.py has none of, a sprite preview, plus the
"Show editor-hidden objects" affordance. Dumb and callback-driven like
TriggerPanel itself (see trigger_fields.py's own module docstring): neither
touches EditHistory or writes anything back on its own. CatalogLineEdit
reports a committed value through its `committed` signal, and TriggerPanel
decides what that means.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from descape import object_catalog, unit_sprites
from descape.value_picker import PickerItem, ValueBrowseDialog, ValueLineEdit

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

HIDDEN_LABEL = "Show editor-hidden objects"


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


def items_for(catalog: Sequence[object_catalog.CatalogEntry]) -> tuple[PickerItem, ...]:
    """catalog's own CatalogEntry rows reduced to value_picker.py's
    domain-free PickerItem shape. `class_id` is dropped -- no widget here
    ever read it."""
    return tuple(
        PickerItem(
            label=entry.name,
            value=entry.id,
            group=entry.category,
            hidden=entry.hidden,
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


class CatalogBrowseDialog(ValueBrowseDialog):
    """A category-grouped tree over one id-dataset catalog, filtered by name
    or id.

    "Show editor-hidden objects" starts unchecked, so `hide_in_editor`
    entries stay out of the tree until opted into.
    """

    def __init__(
        self,
        catalog: Sequence[object_catalog.CatalogEntry],
        default_category: str = "",
        parent=None,
    ):
        self._catalog = catalog
        self._default_category = default_category
        super().__init__(
            items_for(catalog),
            default_group=default_category,
            show_values=False,
            hidden_label=HIDDEN_LABEL,
            preview=catalog_preview,
            title="Browse",
            parent=parent,
        )

    def selected_id(self) -> int | None:
        return self.selected_value()


class CatalogLineEdit(ValueLineEdit):
    """A QLineEdit + type-ahead completer + "..." browse button, for picking
    one id-dataset catalog entry.

    Accepts a typed name (case-insensitively, matched against the catalog) or
    a raw integer, so an id the catalog does not cover -- always possible,
    since it is library-backed until slice 4 -- is still settable.
    """

    def __init__(
        self,
        catalog: Sequence[object_catalog.CatalogEntry],
        default_category: str = "",
        parent=None,
    ):
        self._catalog = catalog
        self._default_category = default_category
        super().__init__(
            items_for(catalog),
            default_group=default_category,
            show_values=False,
            hidden_label=HIDDEN_LABEL,
            preview=catalog_preview,
            browse_title="Browse",
            placeholder="(unset)",
            parent=parent,
        )
