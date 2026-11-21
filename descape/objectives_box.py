"""GH #391: the in-game objectives box, drawn in the game's style.

Two places: in the Objectives dock on its own backdrop (ObjectivesBox), and
over the map's top-right corner, where the game draws it (ObjectivesOverlay). The overlay follows
InstructionOverlay's rules (instruction_overlay.py): a child of MapView, never
of its viewport, sized through panel_scale() and blitted from a cached pixmap.
It is mouse-transparent; a sibling over its checkboxes alone takes clicks.
Taller than the map, it stops at the map's edge and the wheel over it scrolls
its lines instead of zooming. The dock's box sits in a scroll area.

A tick is preview only: the box reports the click and its owner keeps the
ticked set. Nothing here reads or writes the scenario.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from math import ceil
from typing import NamedTuple

from PyQt5.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPen, QPixmap, QRegion, QWheelEvent
from PyQt5.QtWidgets import QApplication, QToolTip, QWidget

from descape import message_markup
from descape.instruction_overlay import panel_scale
from descape.objectives_model import HEADER, OPTIONAL, split_marker

NOTE = "note"
TITLE = "Objectives"
EMPTY_NOTE = "(no objectives on screen)"

# Canvas units, the game's 3840x2160 UI as instruction_overlay.py uses. Not measured from the game's art.
BOX_WIDTH = 900
PAD = 24
TITLE_PX = 48
HEAD_PX = 42
BODY_PX = 38
NOTE_PX = 34
LINE_GAP = 10
CHECK_GAP = 16
# Overlay: the box's offset from the viewport's corner, in widget pixels.
OVERLAY_MARGIN_PX = 8
# Overlay: canvas units scrolled per wheel notch, and the scroll bar's width in widget pixels.
WHEEL_STEP = 120
SCROLL_BAR_PX = 4
SCROLL_BAR = QColor(196, 160, 72, 200)
# In the dock: widget pixels per canvas unit, about 16 px body text.
DOCK_SCALE = 0.42
DOCK_MARGIN_PX = 8

FILL = QColor(28, 28, 32, 215)
FRAME = QColor(150, 150, 160)
TEXT = (250, 250, 245)
NOTE_TEXT = QColor(150, 150, 150)
CHECK_FRAME = QColor(196, 160, 72)
CHECK_INNER = QColor(110, 86, 36)
CHECK_FILL = QColor(20, 18, 14)
TICK = QColor(240, 200, 90)
ROUND_FRAME = QColor(205, 205, 210)
ROUND_INNER = QColor(90, 90, 96)
BACKDROP = (QColor(18, 20, 38), QColor(6, 6, 14))


class BoxEntry(NamedTuple):
    """One line: HEADER, OBJECTIVE or NOTE, its raw text (colour tags kept),
    the trigger index it stands for (-1 for a note) and a tooltip note.
    layout() fills `marker` from the text's leading + or - and drops it."""

    kind: str
    text: str
    index: int = -1
    note: str = ""
    marker: str | None = None


def serif_font(px: int, italic: bool = False) -> QFont:
    font = QFont("Noto Serif")
    font.setFamilies(["Noto Serif", "Times New Roman", "serif"])
    font.setStyleHint(QFont.Serif)
    font.setPixelSize(px)
    font.setBold(True)
    font.setItalic(italic)
    return font


class _Layout(NamedTuple):
    """Canvas-unit geometry: each line's (entry, text rect, checkbox rect or None) and the box's height."""

    lines: list[tuple[BoxEntry, QRectF, QRectF | None, QFont]]
    height: float


def _font_for(kind: str) -> QFont:
    if kind == TITLE:
        return serif_font(TITLE_PX)
    if kind == HEADER:
        return serif_font(HEAD_PX)
    if kind == NOTE:
        return serif_font(NOTE_PX, italic=True)
    return serif_font(BODY_PX)


def layout(entries: Sequence[BoxEntry], width: float) -> _Layout:
    """Where every line goes in a box `width` canvas units wide, the title first."""
    lines = []
    y = PAD
    shown = list(entries) or [BoxEntry(NOTE, EMPTY_NOTE)]
    tags = message_markup.text_colors()
    for entry in [BoxEntry(TITLE, TITLE), *shown]:
        # Headers take no marker: the game's own headers carry none, and a "+" header is unconfirmed.
        if entry.kind not in (TITLE, NOTE, HEADER):
            marker, text = split_marker(entry.text, tags)
            entry = entry._replace(text=text, marker=marker)  # noqa: PLW2901
        font = _font_for(entry.kind)
        height = QFontMetricsF(font).height() + LINE_GAP
        x = PAD
        check = None
        if entry.marker is not None:
            side = BODY_PX * 0.8
            check = QRectF(x, y + (height - side) / 2, side, side)
            x += side + CHECK_GAP
        lines.append((entry, QRectF(x, y, max(width - PAD - x, 1), height), check, font))
        y += height
    return _Layout(lines, y + PAD)


def _paint_check(painter: QPainter, rect: QRectF, ticked: bool) -> None:
    painter.setPen(QPen(CHECK_FRAME, 5))
    painter.setBrush(CHECK_FILL)
    painter.drawRect(rect)
    painter.setPen(QPen(CHECK_INNER, 2))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(rect.adjusted(6, 6, -6, -6))
    if ticked:
        _paint_tick(painter, rect)


def _paint_round(painter: QPainter, rect: QRectF, ticked: bool) -> None:
    """The optional objective's marker: a silver ring."""
    painter.setPen(QPen(ROUND_FRAME, 5))
    painter.setBrush(CHECK_FILL)
    painter.drawEllipse(rect)
    painter.setPen(QPen(ROUND_INNER, 2))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(rect.adjusted(6, 6, -6, -6))
    if ticked:
        _paint_tick(painter, rect)


def _paint_tick(painter: QPainter, rect: QRectF) -> None:
    side = rect.width()
    painter.setPen(QPen(TICK, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawPolyline(*[
        rect.topLeft() + QPointF(side * fx, side * fy) for fx, fy in ((0.2, 0.5), (0.42, 0.75), (0.85, 0.15))
    ])


# How a <name> tag naming a trigger variable reads in the preview. Variables start at 0 in the game.
VARIABLE_NAME, VARIABLE_VALUE, VARIABLE_BOTH = "name", "value", "both"
VARIABLE_START = 0


def variable_text(name: str, mode: str) -> str:
    """A variable tag as the preview shows it: as typed, its starting value, or both."""
    if mode == VARIABLE_VALUE:
        return str(VARIABLE_START)
    if mode == VARIABLE_BOTH:
        return f"{VARIABLE_START} ({name})"
    return f"<{name}>"


def _paint_text(painter: QPainter, entry: BoxEntry, rect: QRectF, font: QFont, colors, state: _BoxState) -> None:
    """The text's colour runs left to right, the run that overflows elided."""
    painter.setFont(font)
    metrics = QFontMetricsF(font)
    if entry.kind == NOTE:
        painter.setPen(NOTE_TEXT)
        painter.drawText(rect, Qt.AlignVCenter, metrics.elidedText(entry.text, Qt.ElideRight, rect.width()))
        return
    text = " ".join(entry.text.splitlines())
    if entry.kind == TITLE:
        runs = [message_markup.Run(text, None, "")]
    else:
        runs = message_markup.parse(text, state.variables, colors)
    x = rect.left()
    for run in runs:
        piece = variable_text(run.text, state.variable_mode) if run.kind == message_markup.VARIABLE else run.text
        room = rect.right() - x
        width = metrics.horizontalAdvance(piece)
        elided = width > room
        if elided:
            piece = metrics.elidedText(piece, Qt.ElideRight, room)
            width = metrics.horizontalAdvance(piece)
        painter.setPen(QColor(*(run.color or TEXT)))
        painter.drawText(QRectF(x, rect.top(), max(width, 1) + 1, rect.height()), Qt.AlignVCenter, piece)
        x += width
        if elided or x >= rect.right():
            break


def paint_box(painter: QPainter, geometry: _Layout, width: float, state: _BoxState) -> None:
    """The box in canvas units at (0, 0): fill, frame, title, lines."""
    box = QRectF(0, 0, width, geometry.height)
    painter.setPen(QPen(FRAME, 3))
    painter.setBrush(FILL)
    painter.drawRect(box.adjusted(1.5, 1.5, -1.5, -1.5))
    colors = message_markup.text_colors()
    for entry, rect, check, font in geometry.lines:
        if check is not None:
            (_paint_round if entry.marker == OPTIONAL else _paint_check)(painter, check, entry.index in state.ticked)
        _paint_text(painter, entry, rect, font, colors, state)


class _BoxState:
    """What both widgets share: the lines, the ticked set and the tick callback."""

    def __init__(self) -> None:
        self.entries: list[BoxEntry] = []
        self.ticked: frozenset[int] = frozenset()
        self.on_tick: Callable[[int], None] = lambda index: None
        # The document's named trigger variables, and how a tag naming one reads (variable_text()).
        self.variables: tuple[str, ...] = ()
        self.variable_mode = VARIABLE_NAME


class ObjectivesBox(QWidget):
    """The dock's preview: the box on its own backdrop, filling the widget's
    width, at least as tall as its lines (its scroll area scrolls the rest)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = _BoxState()
        self.setMinimumSize(160, 80)
        self.setMouseTracking(False)

    def set_entries(
        self, entries: Sequence[BoxEntry], ticked: Collection[int], variables: Sequence[str] = (), mode: str = VARIABLE_NAME
    ) -> None:
        self.state.entries = list(entries)
        self.state.ticked = frozenset(ticked)
        self.state.variables = tuple(variables)
        self.state.variable_mode = mode
        self._fit_height()
        self.updateGeometry()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_height()

    def _fit_height(self) -> None:
        needed = ceil(layout(self.state.entries, self._canvas_width()).height * DOCK_SCALE) + 2 * DOCK_MARGIN_PX
        if self.minimumHeight() != needed:
            self.setMinimumHeight(needed)

    def entries(self) -> list[BoxEntry]:
        return list(self.state.entries)

    def _canvas_width(self) -> float:
        return max(self.width() - 2 * DOCK_MARGIN_PX, 1) / DOCK_SCALE

    def check_rects(self) -> list[tuple[int, QRectF]]:
        """(trigger index, widget-pixel checkbox rect) per objective line."""
        geometry = layout(self.state.entries, self._canvas_width())
        return [
            (entry.index, _to_widget(check, DOCK_SCALE, DOCK_MARGIN_PX, DOCK_MARGIN_PX))
            for entry, _rect, check, _font in geometry.lines
            if check is not None
        ]

    def sizeHint(self) -> QSize:
        geometry = layout(self.state.entries, BOX_WIDTH)
        return QSize(round(BOX_WIDTH * DOCK_SCALE) + 2 * DOCK_MARGIN_PX, ceil(geometry.height * DOCK_SCALE) + 2 * DOCK_MARGIN_PX)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, BACKDROP[0])
        gradient.setColorAt(1, BACKDROP[1])
        painter.fillRect(self.rect(), gradient)
        painter.translate(DOCK_MARGIN_PX, DOCK_MARGIN_PX)
        painter.scale(DOCK_SCALE, DOCK_SCALE)
        width = self._canvas_width()
        paint_box(painter, layout(self.state.entries, width), width, self.state)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            for index, rect in self.check_rects():
                if rect.contains(QPointF(event.pos())):
                    self.state.on_tick(index)
                    return
        super().mousePressEvent(event)

    def event(self, event) -> bool:
        if event.type() == QEvent.ToolTip:
            note = self._note_at(QPointF(event.pos()))
            if note:
                QToolTip.showText(event.globalPos(), note, self)
            else:
                QToolTip.hideText()
                event.ignore()
            return True
        return super().event(event)

    def _note_at(self, pos: QPointF) -> str:
        geometry = layout(self.state.entries, self._canvas_width())
        for entry, rect, _check, _font in geometry.lines:
            line = _to_widget(QRectF(0, rect.top(), self._canvas_width(), rect.height()), DOCK_SCALE, DOCK_MARGIN_PX, DOCK_MARGIN_PX)
            if line.contains(pos):
                return entry.note
        return ""


def _to_widget(rect: QRectF, scale: float, dx: float, dy: float) -> QRectF:
    return QRectF(rect.left() * scale + dx, rect.top() * scale + dy, rect.width() * scale, rect.height() * scale)


class _CheckHits(QWidget):
    """Takes clicks over the overlay's checkboxes only (its mask), so the
    rest of the box lets the map have the mouse."""

    def __init__(self, parent: QWidget, overlay: ObjectivesOverlay) -> None:
        super().__init__(parent)
        self._overlay = overlay
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setCursor(Qt.PointingHandCursor)
        self.hide()

    def wheelEvent(self, event) -> None:
        # Over a checkbox the wheel lands here, not on the map's viewport: scroll a long box, else zoom the map.
        if self._overlay.overflow() > 0:
            self._overlay.scroll_by(-event.angleDelta().y() / 120 * WHEEL_STEP * self._overlay.scale())
            return
        self._overlay.forward_wheel(event, self)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        point = QPointF(event.pos()) + QPointF(self.pos() - self._overlay.pos())
        for index, rect in self._overlay.check_rects():
            if rect.contains(point):
                self._overlay.state.on_tick(index)
                return None
        return super().mousePressEvent(event)


class ObjectivesOverlay(QWidget):
    """The box over the map's top-right corner, at game proportions, clipped
    to the viewport's right half so it clears the centred instruction box."""

    def __init__(self, map_view: QWidget) -> None:
        super().__init__(map_view)
        self.state = _BoxState()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.NoFocus)
        self.hide()
        self._hits = _CheckHits(map_view, self)
        self._area = QRect()
        self._scale = 0.0
        self._pixmap = QPixmap()
        self._shown = False
        # Widget pixels scrolled past the top, while the box is taller than the map.
        self._scroll = 0
        # The composed box's height in widget pixels, and its layout width in canvas units.
        self._full_height = 0
        self._canvas_width = float(BOX_WIDTH)
        # The viewport's own Resize: MapView's arrives before its viewport is laid out. Its Wheel too.
        self._viewport = map_view.viewport()
        self._viewport.installEventFilter(self)

    def set_entries(
        self, entries: Sequence[BoxEntry], ticked: Collection[int], variables: Sequence[str] = (), mode: str = VARIABLE_NAME
    ) -> None:
        self.state.entries = list(entries)
        self.state.ticked = frozenset(ticked)
        self.state.variables = tuple(variables)
        self.state.variable_mode = mode
        if self._shown:
            self._refit()

    def entries(self) -> list[BoxEntry]:
        return list(self.state.entries)

    def set_shown(self, shown: bool) -> None:
        self._shown = bool(shown)
        if not shown:
            self.hide()
            self._hits.hide()
            return
        self._refit()

    def shown(self) -> bool:
        return self._shown

    def eventFilter(self, watched, event) -> bool:  # Qt override
        if watched is self._viewport and self._shown:
            if event.type() == QEvent.Resize:
                self._refit()
            elif (
                event.type() == QEvent.Wheel
                and self.overflow() > 0
                and self.geometry().contains(self._viewport.mapToParent(event.pos()))
            ):
                self.scroll_by(-event.angleDelta().y() / 120 * WHEEL_STEP * self._scale)
                return True
        return super().eventFilter(watched, event)

    def overflow(self) -> int:
        """Widget pixels of the box below the map's edge, scrolled or not. From
        the logical height: the pixmap's is rounded up at a fractional scale."""
        return max(self._full_height - self.height(), 0) if self._shown else 0

    def scale(self) -> float:
        return self._scale

    def forward_wheel(self, event, source: QWidget) -> None:
        """Hand a wheel event that landed on `source` to the map's viewport, so it zooms as usual."""
        pos = self._viewport.mapFrom(self.parentWidget(), source.mapToParent(event.pos()))
        forwarded = QWheelEvent(
            QPointF(pos), QPointF(event.globalPos()), event.pixelDelta(), event.angleDelta(),
            event.buttons(), event.modifiers(), event.phase(), event.inverted(),
        )
        QApplication.sendEvent(self._viewport, forwarded)

    def scroll_offset(self) -> int:
        return self._scroll

    def scroll_by(self, pixels: float) -> None:
        self._set_scroll(self._scroll + round(pixels))

    def _viewport_rect(self) -> QRect:
        return self._viewport.geometry()

    def check_rects(self) -> list[tuple[int, QRectF]]:
        """(trigger index, this widget's pixel checkbox rect) per objective line."""
        geometry = layout(self.state.entries, self._canvas_width)
        return [
            (entry.index, _to_widget(check, self._scale, 0, -self._scroll))
            for entry, _rect, check, _font in geometry.lines
            if check is not None
        ]

    def _refit(self) -> None:
        self._area = self._viewport_rect()
        self._scale = panel_scale(self._area.width(), self._area.height())
        # Narrower than the right half allows: laid out that wide, so lines elide rather than clip.
        room = max(self._area.width() // 2 - OVERLAY_MARGIN_PX, 1)
        self._canvas_width = min(float(BOX_WIDTH), room / self._scale)
        geometry = layout(self.state.entries, self._canvas_width)
        ratio = self.devicePixelRatioF()
        full = QSize(max(1, round(self._canvas_width * self._scale)), max(1, round(geometry.height * self._scale)))
        self._full_height = full.height()
        self._pixmap = QPixmap(ceil(full.width() * ratio), ceil(full.height() * ratio))
        self._pixmap.setDevicePixelRatio(ratio)
        self._pixmap.fill(Qt.transparent)
        painter = QPainter(self._pixmap)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        painter.scale(self._scale, self._scale)
        paint_box(painter, geometry, self._canvas_width, self.state)
        painter.end()
        height = min(full.height(), max(self._area.height() - 2 * OVERLAY_MARGIN_PX, 1))
        width = min(full.width(), room)
        left = self._area.x() + self._area.width() - OVERLAY_MARGIN_PX - width
        self.setGeometry(left, self._area.y() + OVERLAY_MARGIN_PX, width, height)
        self.show()
        self.raise_()
        self._hits.setGeometry(self.geometry())
        self._set_scroll(self._scroll)

    def _set_scroll(self, scroll: int) -> None:
        """Clamp the scroll, then re-mask the checkbox clicks and repaint."""
        self._scroll = max(0, min(int(scroll), int(self.overflow())))
        self.update()
        region = QRegion()
        for _index, rect in self.check_rects():
            region = region.united(QRegion(rect.toAlignedRect()))
        region = region.intersected(QRegion(QRect(0, 0, self.width(), self.height())))
        if region.isEmpty():
            self._hits.hide()
            return
        self._hits.setMask(region)
        self._hits.show()
        self._hits.raise_()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, -self._scroll, self._pixmap)
        overflow = self.overflow()
        if overflow > 0:
            # A thin bar on the right edge: its length is the share shown, its place the scroll.
            total = self.height() + overflow
            length = max(self.height() * self.height() / total, 12)
            top = (self.height() - length) * self._scroll / overflow
            painter.fillRect(QRectF(self.width() - SCROLL_BAR_PX - 2, top, SCROLL_BAR_PX, length), SCROLL_BAR)
        painter.end()
