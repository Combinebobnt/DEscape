"""GH #167: which way a trigger-created unit faces, under its facet field.

Two effects carry the field: Create Object (`facet`, unit
`object_list_unit_id`) and Replace Object (`facet2`, unit
`object_list_unit_id_2`). The preview is observe-only: bind() follows the
property form's own row widgets and nothing here writes the entry.

What a facet number means is not measured yet. The working hypothesis is
facet_rotation()'s: facet k is the same radians a placed unit's `rotation`
holds for facing k of 16. It lives in that one function, so the in-game check
can change it in one place.
"""

from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, QSize, Qt
from PyQt5.QtGui import QPainter, QPalette, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QAbstractSpinBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from descape import constant_picker, iso_geometry, object_catalog, unit_rotation
from descape.value_picker import ValueLineEdit

FACET_STEPS = 16
UNSET = -1
TILE_PX = 64
# Any tile_px gives the same 2:1 ratio; half_dims() is what keeps it exact.
_HALF_W, _HALF_H = iso_geometry.half_dims(TILE_PX)
# The arrow reaches this share of the way from the centre to the diamond's edge.
_ARROW_REACH = 0.8
_HEAD_PX = 7.0


def facet_rotation(facet: int) -> float | None:
    """Stored radians for a facet, or None when it is negative (the game picks one).
    Facet k is taken as facing k of FACET_STEPS, wrapping past it."""
    if facet < 0:
        return None
    return unit_rotation.facing_to_rotation(facet, FACET_STEPS)


def arrow_vector(rotation: float) -> tuple[float, float]:
    """World facing (cos r, sin r) on screen (y down), in pixels of a TILE_PX tile.

    iso_geometry.tile_screen_origin() puts sx at (x+y)*half_w and sy at
    (y-x)*half_h, so world +x reads up-right and +y down-right.
    """
    wx, wy = math.cos(rotation), math.sin(rotation)
    return ((wx + wy) * _HALF_W, (wy - wx) * _HALF_H)


def screen_angle_deg(dx: float, dy: float) -> float:
    """A screen vector's angle clockwise from screen-right, in [0, 360)."""
    return math.degrees(math.atan2(dy, dx)) % 360.0


class _ArrowTile(QWidget):
    """One iso tile with an arrow from its centre along arrow_vector()."""

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.NoFocus)
        self.setFixedSize(2 * _HALF_W + 4, 2 * _HALF_H + 4)
        self.rotation: float | None = None

    def set_rotation(self, rotation: float | None) -> None:
        self.rotation = rotation
        self.update()

    def arrow_points(self) -> tuple[QPointF, QPointF]:
        """(centre, tip) in widget pixels. Halved: the tile's edge is half a world step out."""
        centre = QPointF(self.width() / 2, self.height() / 2)
        dx, dy = arrow_vector(self.rotation or 0.0)
        reach = 0.5 * _ARROW_REACH
        return centre, QPointF(centre.x() + dx * reach, centre.y() + dy * reach)

    def paintEvent(self, _event) -> None:
        palette = self.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        diamond = QPolygonF(
            [QPointF(cx - _HALF_W, cy), QPointF(cx, cy - _HALF_H), QPointF(cx + _HALF_W, cy), QPointF(cx, cy + _HALF_H)]
        )
        painter.setPen(QPen(palette.color(QPalette.Mid), 1))
        painter.setBrush(palette.color(QPalette.Base))
        painter.drawPolygon(diamond)
        if self.rotation is None:
            return
        centre, tip = self.arrow_points()
        color = palette.color(QPalette.Highlight)
        painter.setPen(QPen(color, 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(centre, tip)
        length = math.hypot(tip.x() - centre.x(), tip.y() - centre.y()) or 1.0
        ux, uy = (tip.x() - centre.x()) / length, (tip.y() - centre.y()) / length
        back = QPointF(tip.x() - ux * _HEAD_PX, tip.y() - uy * _HEAD_PX)
        side = QPointF(-uy * _HEAD_PX / 2, ux * _HEAD_PX / 2)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([tip, back + side, back - side]))


class FacetPreview(QWidget):
    """The arrow tile, the unit's sprite at that facing (install only) and a note."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.NoFocus)
        self.tile = _ArrowTile()
        self.sprite_label = QLabel()
        self.sprite_label.setFocusPolicy(Qt.NoFocus)
        self.sprite_label.setAlignment(Qt.AlignCenter)
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        self.note_label.setMinimumWidth(1)
        self.note_label.setFocusPolicy(Qt.NoFocus)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.tile, 0, Qt.AlignVCenter)
        row.addWidget(self.sprite_label, 0, Qt.AlignVCenter)
        row.addStretch(1)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 2, 0, 2)
        column.setSpacing(2)
        column.addLayout(row)
        column.addWidget(self.note_label)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.rotation: float | None = None
        self._shown: bool | None = None
        self._layout: tuple | None = None
        self._fields: dict = {}
        self._facet_name = ""
        self._unit_name = ""
        self._read = None
        self._entry = None
        self._on_resize = None
        self._pixmaps: dict[tuple[int, int], QPixmap | None] = {}

    def minimumSizeHint(self) -> QSize:
        return QSize(self.tile.width(), super().minimumSizeHint().height())

    # -- wiring --------------------------------------------------------------

    def bind(self, fields: dict, facet_name: str, unit_name: str, read, entry=None, on_resize=None) -> None:
        """Follow the facet and unit rows of `fields` (field name -> row
        widget). `entry` is the one entry the form shows, or None for a
        multi-entry form; `on_resize` runs when the row shows or hides."""
        self._fields = {name: fields[name] for name in (facet_name, unit_name) if name in fields}
        self._facet_name = facet_name
        self._unit_name = unit_name
        self._read = read
        self._entry = entry
        self._on_resize = on_resize
        facet = self._fields.get(facet_name)
        if isinstance(facet, QAbstractSpinBox):
            # Bound methods of this QObject: the connections die with it.
            facet.valueChanged.connect(self._field_changed)
        unit = self._fields.get(unit_name)
        if isinstance(unit, ValueLineEdit):
            unit.committed.connect(self._field_changed)
        self.refresh()

    def detach(self) -> None:
        """Stop following: the rows are being torn down with this preview."""
        self._fields = {}
        self._entry = None
        self._on_resize = None

    def _field_changed(self, *_args) -> None:
        self.refresh()

    def _facet(self) -> int | None:
        """The live facet, or None when the spinbox shows a differing selection."""
        widget = self._fields.get(self._facet_name)
        if isinstance(widget, QAbstractSpinBox):
            if getattr(widget, "is_indeterminate", lambda: False)():
                return None
            return int(widget.value())
        value = None if self._entry is None else self._read(self._entry, self._facet_name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _unit(self) -> int:
        """The live unit const, UNSET when unset, differing or unreadable."""
        widget = self._fields.get(self._unit_name)
        if isinstance(widget, ValueLineEdit):
            value = widget.value()
        else:
            value = None if self._entry is None else self._read(self._entry, self._unit_name)
        return value if isinstance(value, int) and not isinstance(value, bool) else UNSET

    # -- rendering -----------------------------------------------------------

    def refresh(self) -> None:
        if not self._fields:
            return
        facet = self._facet()
        rotation = None if facet is None else facet_rotation(facet)
        shown = rotation is not None
        self.rotation = rotation
        if shown:
            self._show(facet, rotation, self._unit())
        if shown != self._shown:
            # Explicit even the first time: a never-shown row would be shown by the form's fit pass.
            self.setVisible(shown)
        layout = (shown, self.arrow_shown, self.sprite_shown)
        if layout != self._layout:
            self._shown, self._layout = shown, layout
            self.updateGeometry()
            if self._on_resize is not None:
                self._on_resize()

    def _show(self, facet: int, rotation: float, unit: int) -> None:
        known = unit >= 0 and object_catalog.is_known_object(unit)
        if known and not unit_rotation.rotation_is_angle(unit):
            self.tile.set_rotation(None)
            self.tile.hide()
            self._set_sprite(None)
            self.note_label.setText(f"{object_catalog.display_name(unit)} has no facing")
            self.note_label.show()
            return
        self.tile.set_rotation(rotation)
        self.tile.show()
        self.note_label.clear()
        self.note_label.hide()
        self._set_sprite(self._pixmap(unit, facet, rotation) if known else None)

    def _pixmap(self, unit: int, facet: int, rotation: float) -> QPixmap | None:
        """Cached per (unit, facet), so stepping the spinbox decodes each frame once."""
        key = (unit, facet % FACET_STEPS)
        if key not in self._pixmaps:
            self._pixmaps[key] = constant_picker.preview_pixmap(unit, rotation)
        return self._pixmaps[key]

    def _set_sprite(self, pixmap: QPixmap | None) -> None:
        if pixmap is None:
            self.sprite_label.clear()
            self.sprite_label.hide()
        else:
            self.sprite_label.setPixmap(pixmap)
            self.sprite_label.show()

    @property
    def sprite_shown(self) -> bool:
        return not self.sprite_label.isHidden()

    @property
    def arrow_shown(self) -> bool:
        return not self.tile.isHidden() and self.tile.rotation is not None
