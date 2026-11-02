"""TASK-550: one Display Instructions message drawn over the map at game proportions.

Observe-only, and fixed to the viewport as the game's panel is fixed to the
screen. A child of MapView, never of its viewport: QWidget.scroll() carries the
viewport's children along with the map. Not a scene item either, which would
land in scene.render() captures, and not drawForeground(), whose pixels
accelerated scrolling moves.
"""

from __future__ import annotations

from math import ceil
from typing import NamedTuple

from PyQt5.QtCore import QRect, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPixmap, QTextDocument
from PyQt5.QtWidgets import QWidget

from descape.viewer_canvas import map_overlay_font

# Canvas units: the game's 3840x2160 UI, from widgetui/instructionspanel.json. Art colours
# measured from widgetui/textures/ingame/panels/npanel_*_4k.png. tests/test_instruction_overlay.py checks both.
CANVAS_HEIGHT = 2160
PANEL_TOP = 300
# TOP, MIDDLE, BOTTOM. The one place to correct if the game disagrees (the library's docstrings do).
SLOT_OFFSETS = (0, 160, 320)
PANEL_WIDTH = 1000
ICON_RECT = (10, 10, 140, 140)
TEXT_RECT = (160, 10, 820, 200)
FONT_PX = 42

# Guesses: at least one slot tall, grows with its text, clips past the Surround's height.
PANEL_MIN_HEIGHT = 160
PANEL_MAX_HEIGHT = 450
ICON_FRAME = 4

MIN_SCALE = 0.3  # 12.6 px text
WIDTH_MARGIN_PX = 8

ART_ALPHA = 191
ART_FILL = (24, 21, 18)
# (inset, thickness, colour), outermost first.
ART_RINGS = (
    (0, 1, (32, 25, 25)),
    (1, 2, (147, 147, 147)),
    (3, 1, (15, 13, 11)),
)

ICON_PX = ICON_RECT[2]


class InstructionView(NamedTuple):
    """What crosses from the trigger panel to the map."""

    html: str
    slot: int
    icon: QPixmap | None
    tint: tuple[int, int, int] | None


def panel_scale(width: int, height: int) -> float:
    """Widget pixels per canvas unit in a `width` x `height` viewport."""
    scale = max(height / CANVAS_HEIGHT, MIN_SCALE)
    return min(scale, max(width - 2 * WIDTH_MARGIN_PX, 1) / PANEL_WIDTH)


def panel_top(height: int, slot: int) -> int:
    """The box's top in viewport pixels: the slot's share of the height, whatever the scale's floor does."""
    return round(height * (PANEL_TOP + SLOT_OFFSETS[slot]) / CANVAS_HEIGHT)


def _ring(rect: QRectF, thickness: float) -> QPainterPath:
    path = QPainterPath()  # the default odd-even fill leaves the inner rect out
    path.addRect(rect)
    path.addRect(rect.adjusted(thickness, thickness, -thickness, -thickness))
    return path


class InstructionOverlay(QWidget):
    """The box alone, sized to it. Text is laid out in canvas units and painted
    through a scale, so line breaks do not change with the window."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.NoFocus)
        self.hide()
        self._view: InstructionView | None = None
        self._area = QRect()
        self._scale = 0.0
        self._ratio = 0.0
        self._box = QSize()
        self._pixmap = QPixmap()

    def view(self) -> InstructionView | None:
        return self._view

    def set_view(self, view: InstructionView | None) -> None:
        self._view = view
        if view is None:
            self.hide()
            return
        self._compose()
        self._fit()
        self.show()

    def place(self, viewport_rect: QRect) -> None:
        """Follow the viewport, in this widget's parent's coordinates."""
        self._area = QRect(viewport_rect)
        if self._view is None:
            return
        scale = panel_scale(self._area.width(), self._area.height())
        if scale != self._scale or self.devicePixelRatioF() != self._ratio:
            self._compose()
        self._fit()

    def _fit(self) -> None:
        area, box = self._area, self._box
        left = area.x() + (area.width() - box.width()) // 2
        self.setGeometry(left, area.y() + panel_top(area.height(), self._view.slot), box.width(), box.height())
        self.raise_()

    def _compose(self) -> None:
        view = self._view
        self._scale = panel_scale(self._area.width(), self._area.height())
        self._ratio = self.devicePixelRatioF()
        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(map_overlay_font(FONT_PX))
        text_x, text_y, text_width, _height = TEXT_RECT
        document.setTextWidth(text_width)
        document.setHtml(view.html)
        height = min(max(ceil(document.size().height()) + 2 * text_y, PANEL_MIN_HEIGHT), PANEL_MAX_HEIGHT)

        self._box = QSize(max(1, round(PANEL_WIDTH * self._scale)), max(1, round(height * self._scale)))
        self._pixmap = QPixmap(ceil(self._box.width() * self._ratio), ceil(self._box.height() * self._ratio))
        self._pixmap.setDevicePixelRatio(self._ratio)
        self._pixmap.fill(Qt.transparent)
        painter = QPainter(self._pixmap)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        painter.scale(self._scale, self._scale)
        self._paint_art(painter, height)
        if view.icon is not None:
            self._paint_icon(painter, view)
        painter.translate(text_x, text_y)
        document.drawContents(painter, QRectF(0, 0, text_width, height - text_y))
        painter.end()

    @staticmethod
    def _paint_art(painter: QPainter, height: int) -> None:
        box = QRectF(0, 0, PANEL_WIDTH, height)
        painter.setCompositionMode(QPainter.CompositionMode_Source)  # replaces the fill, as in the art
        painter.fillRect(box, QColor(*ART_FILL, ART_ALPHA))
        for inset, thickness, color in ART_RINGS:
            painter.fillPath(_ring(box.adjusted(inset, inset, -inset, -inset), thickness), QColor(*color, ART_ALPHA))
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

    @staticmethod
    def _paint_icon(painter: QPainter, view: InstructionView) -> None:
        left, top, width, height = ICON_RECT
        icon = view.icon
        painter.drawPixmap(left + (width - icon.width()) // 2, top + (height - icon.height()) // 2, icon)
        if view.tint is not None:
            painter.fillPath(_ring(QRectF(left, top, width, height), ICON_FRAME), QColor(*view.tint))

    def paintEvent(self, _event) -> None:
        # Qt repaints this for any dirty region under it, so it only blits.
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.end()
