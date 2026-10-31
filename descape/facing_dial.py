"""TASK-023: a radial dial beside the unit inspector's Rotation spinbox.

Spoke k points where facing k's sprite faces on screen, in the map's own
projection: the iso 2:1 squash whenever the map renders isometric, a plain
circle in Flat. An all-gate selection gets four orientation axes instead.

The dial never writes anything itself. The spinbox beside it stays the value
holder and the only emitter: a spoke commits through `on_commit` (the
spinbox's setValue) and the wheel goes through `on_step` (its stepBy), so
every edit takes the spinbox's existing path to the model.

Facings are live (TASK-548): a drag commits each spoke it crosses, and
`on_drag` brackets the press and release so the viewer can fold the spokes
into one undo step. A gate's axis swaps its const and moves its footprint, so
gate mode previews on the dial and commits once, on release.

The geometry functions at the top are Qt-free so they test on their own.
"""

from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, QSize, Qt
from PyQt5.QtGui import QPainter, QPalette, QPen, QPolygonF
from PyQt5.QtWidgets import QSizePolicy, QWidget

from descape import facet_preview, gate_orientation, iso_geometry, unit_rotation, unit_sprites

FACING = "facing"
GATE = "gate"
GATE_AXES = 4

# Any tile_px gives the same 2:1 ratio; the arrow and its inverse must share it.
_HALF_W, _HALF_H = iso_geometry.half_dims(facet_preview.TILE_PX)
# The longest screen vector arrow_vector() gives for a unit world vector (at 45 degrees).
_ISO_REACH = math.sqrt(2) * _HALF_W

DIAL_WIDTH = 96
DIAL_HEIGHT = 64
_MARGIN_PX = 4
# Presses this close to the centre have no direction, so they pick nothing.
_DEAD_ZONE_PX = 4
_POINTER_REACH = 0.85
_TICK_INNER = 0.8
_HEAD_PX = 6.0
_WHEEL_NOTCH = 120


def screen_vector(rotation: float, flat: bool) -> tuple[float, float]:
    """Unit-length screen direction (y down) a stored `rotation` faces."""
    if flat:
        angle = rotation + math.radians(unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG)
        return math.cos(angle), math.sin(angle)
    dx, dy = facet_preview.arrow_vector(rotation)
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def ring_point(rotation: float, flat: bool) -> tuple[float, float]:
    """Where `rotation` meets the dial's ring: the unit circle in Flat, the
    iso image of a world circle (semi-axes 1 and 0.5) otherwise."""
    if flat:
        return screen_vector(rotation, True)
    dx, dy = facet_preview.arrow_vector(rotation)
    return dx / _ISO_REACH, dy / _ISO_REACH


def rotation_at(dx: float, dy: float, flat: bool) -> float:
    """screen_vector()'s inverse: the world rotation, in radians, a screen direction faces."""
    if flat:
        return math.atan2(dy, dx) - math.radians(unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG)
    wx = (dx / _HALF_W - dy / _HALF_H) / 2
    wy = (dx / _HALF_W + dy / _HALF_H) / 2
    return math.atan2(wy, wx)


def facing_at(dx: float, dy: float, count: int, flat: bool) -> int:
    """The nearest of `count` facings to a screen direction."""
    return unit_rotation.rotation_to_facing(rotation_at(dx, dy, flat), count)


def gate_rotation(index: int) -> float:
    """World angle, in radians, that gate orientation `index` runs along."""
    return math.radians(index * gate_orientation.RUN_ANGLE_STEP_DEG)


def gate_at(dx: float, dy: float, flat: bool) -> int:
    """The gate orientation axis nearest a screen direction; either end of an axis picks it."""
    steps = math.degrees(rotation_at(dx, dy, flat)) / gate_orientation.RUN_ANGLE_STEP_DEG
    return math.floor(steps + 0.5) % GATE_AXES


def tick_rotations(count: int) -> list[float]:
    """Tick marks for a `count`-facing ring: every facing up to 16, every 2nd up
    to 32, the 8 compass points above that. Clicks still snap to all `count`."""
    if count <= 16:
        return [unit_rotation.facing_to_rotation(k, count) for k in range(count)]
    if count <= 32:
        return [unit_rotation.facing_to_rotation(k, count) for k in range(0, count, 2)]
    return [k * math.pi / 4 for k in range(8)]


class FacingDial(QWidget):
    """The dial widget. `on_commit(value)` per spoke a facing drag crosses, or once
    per gate click or drag, and `on_step(+-1)` per wheel notch.

    `on_drag(True)` opens a facing drag at the press, and False means it was
    refused and the viewer has said why. `on_drag(False)` closes it at the
    release, or `on_drag(False, hidden=True)` when the dial hid first, since a
    hidden widget never gets its release. `_drag_open` decides what a move or
    release does, not `mode`: a re-populate can change that mid-drag. Changing
    to gates closes it the same way, and the rest of that press does nothing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(DIAL_WIDTH, DIAL_HEIGHT)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # The keyboard stays on the spinbox beside it.
        self.setFocusPolicy(Qt.NoFocus)
        self.setAccessibleName("Rotation dial")
        self.mode = FACING
        self.count = 16
        self.value: int | None = 0
        self.flat = False
        self._drag_value: int | None = None
        self._drag_open = False
        self._press_mode = FACING
        self._wheel_accum = 0
        self.on_commit = lambda _value: None
        self.on_step = lambda _steps: None
        self.on_drag = lambda _active, hidden=False: True

    def sizeHint(self) -> QSize:
        return QSize(DIAL_WIDTH, DIAL_HEIGHT)

    # -- state ---------------------------------------------------------------

    def set_mode(self, mode: str, count: int) -> None:
        if mode not in (FACING, GATE):
            raise ValueError(f"unknown dial mode {mode!r}")
        if mode != FACING and self._drag_open:
            self._drag_open = False
            self._drag_value = None
            self.on_drag(False, hidden=True)
        self.mode = mode
        self.count = GATE_AXES if mode == GATE else count
        self.update()

    def set_value(self, value: int | None) -> None:
        """The spinbox's value, or None for a "(mixed)" group: ring and ticks, no pointer."""
        self.value = value
        self.update()

    def set_flat(self, flat: bool) -> None:
        self.flat = flat
        self.update()

    @property
    def shown_value(self) -> int | None:
        """What the pointer shows: a live drag's preview, else the value."""
        return self._drag_value if self._drag_value is not None else self.value

    # -- geometry ------------------------------------------------------------

    def _centre(self) -> QPointF:
        return QPointF(self.width() / 2, self.height() / 2)

    def _radius(self) -> float:
        """Horizontal ring radius in px; the iso ring is half as tall."""
        if self.flat:
            return min(self.width(), self.height()) / 2 - _MARGIN_PX
        return min(self.width() / 2, self.height()) - _MARGIN_PX

    def _rotation_of(self, value: int) -> float:
        if self.mode == GATE:
            return gate_rotation(value)
        return unit_rotation.facing_to_rotation(value, self.count)

    def _point(self, rotation: float, reach: float) -> QPointF:
        ux, uy = ring_point(rotation, self.flat)
        centre, radius = self._centre(), self._radius() * reach
        return QPointF(centre.x() + ux * radius, centre.y() + uy * radius)

    def spoke_point(self, value: int, reach: float = _POINTER_REACH) -> QPointF:
        """Widget pixel on facing (or gate axis) `value`'s spoke."""
        return self._point(self._rotation_of(value), reach)

    def value_at(self, x: float, y: float) -> int | None:
        """The facing or gate axis under a widget pixel, or None at the centre."""
        centre = self._centre()
        dx, dy = x - centre.x(), y - centre.y()
        if math.hypot(dx, dy) < _DEAD_ZONE_PX or self.count <= 1:
            return None
        if self.mode == GATE:
            return gate_at(dx, dy, self.flat)
        return facing_at(dx, dy, self.count, self.flat)

    # -- input ---------------------------------------------------------------

    def _commit_spoke(self, value: int | None) -> None:
        """A live drag reaching `value`; a spoke it already committed writes nothing, so
        a Ctrl+Z or Rotate key during the drag is not overwritten by the release."""
        if value is None or value == self._drag_value:
            return
        self._drag_value = value
        self.update()
        self.on_commit(value)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        event.accept()
        value = self.value_at(event.x(), event.y())
        self._press_mode = self.mode
        # A release lost to a modal leaves both behind, so this press starts afresh.
        self._drag_open = False
        self._drag_value = None if self.mode == FACING else value
        self.update()
        if self.mode == FACING:
            self._drag_open = self.on_drag(True)
            if self._drag_open:
                self._commit_spoke(value)

    def mouseMoveEvent(self, event) -> None:
        if not event.buttons() & Qt.LeftButton:
            event.ignore()
            return
        event.accept()
        value = self.value_at(event.x(), event.y())
        if self._drag_open:
            self._commit_spoke(value)
        elif self.mode != FACING and self._press_mode != FACING and value is not None:
            self._drag_value = value
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        event.accept()
        value = self.value_at(event.x(), event.y())
        if self._drag_open:
            self._commit_spoke(value)
            self._drag_value = None
            self.update()
            self._drag_open = False
            self.on_drag(False)
            return
        if FACING in (self.mode, self._press_mode):
            return  # a refused press, or a late release after a hide or mode change closed the drag
        if value is None:
            value = self._drag_value
        self._drag_value = None
        self.update()
        if value is not None:
            # One commit per press-release, so a gate turn is one undo step.
            self.on_commit(value)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._drag_value = None
        if self._drag_open:
            self._drag_open = False
            self.on_drag(False, hidden=True)

    def wheelEvent(self, event) -> None:
        # Through the spinbox's own stepBy: wrap, "(mixed)" and undo match it exactly.
        self._wheel_accum += event.angleDelta().y()
        while abs(self._wheel_accum) >= _WHEEL_NOTCH:
            step = 1 if self._wheel_accum > 0 else -1
            self._wheel_accum -= step * _WHEEL_NOTCH
            self.on_step(step)
        event.accept()

    # -- paint ---------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        palette = self.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        centre, radius = self._centre(), self._radius()
        painter.setPen(QPen(palette.color(QPalette.Mid), 1))
        painter.setBrush(palette.color(QPalette.Base))
        painter.drawEllipse(centre, radius, radius if self.flat else radius / 2)
        highlight = palette.color(QPalette.Highlight)
        shown = self.shown_value
        if self.mode == GATE:
            for index in range(GATE_AXES):
                rotation = gate_rotation(index)
                if index == shown:
                    painter.setPen(QPen(highlight, 3, Qt.SolidLine, Qt.RoundCap))
                else:
                    painter.setPen(QPen(palette.color(QPalette.Mid), 1))
                painter.drawLine(self._point(rotation, 1.0), self._point(rotation + math.pi, 1.0))
            return
        painter.setPen(QPen(palette.color(QPalette.WindowText), 1))
        for rotation in tick_rotations(self.count):
            painter.drawLine(self._point(rotation, _TICK_INNER), self._point(rotation, 1.0))
        if shown is None:
            return
        tip = self.spoke_point(shown)
        painter.setPen(QPen(highlight, 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(centre, tip)
        length = math.hypot(tip.x() - centre.x(), tip.y() - centre.y()) or 1.0
        ux, uy = (tip.x() - centre.x()) / length, (tip.y() - centre.y()) / length
        back = QPointF(tip.x() - ux * _HEAD_PX, tip.y() - uy * _HEAD_PX)
        side = QPointF(-uy * _HEAD_PX / 2, ux * _HEAD_PX / 2)
        painter.setBrush(highlight)
        painter.drawPolygon(QPolygonF([tip, back + side, back - side]))
