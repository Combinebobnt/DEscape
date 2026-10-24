"""SwatchGridView: a grid of captioned swatches under group headings (GH #161).

Domain-free like value_picker.py: it takes PickerItems and a
`swatch(value) -> QPixmap` hook and imports nothing terrain-side, so
terrain_browser.py hosts it as the Grid half of its picker.

One QListWidget in IconMode with no grid size. A grid size would clip each
heading to one cell; without it a heading's size hint can span the viewport,
so it takes a whole row and the cells after it start a new one. Headings carry
Qt.NoItemFlags, which keeps arrow keys and clicks from ever making one current.

The armed value is stored here, not read from Qt's selection, which can go
empty while a value is armed (a Ctrl+click on the selected cell, a press
dragged onto a heading). The delegate paints every cell and the armed cell's
frame by hand and never looks at State_Selected.

Swatches fill lazily: nothing is asked of the hook before the first show, and
a 0 ms timer then fills cells in the viewport first, the rest after, about
10 ms per tick. It stops when nothing is left and while hidden, and backs off
while a mouse button is held, so a decode never lands inside a Draw stroke.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence

from PyQt5.QtCore import QEvent, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics, QIcon, QPalette, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QListView,
    QListWidget,
    QListWidgetItem,
    QStyledItemDelegate,
    QWidget,
)

from descape import item_activation
from descape.value_picker import PickerItem

_HEADING_ROLE = Qt.UserRole + 1
_FILLED_ROLE = Qt.UserRole + 2

PAD = 4  # inside a cell, around the swatch and the caption
GAP = 2  # between the swatch and its caption
SPACING = 2  # between cells (QListView.spacing)
FRAME_PX = 2  # the armed cell's Highlight frame
CAPTION_SCALE = 0.8
CAPTION_MIN_PT = 7.0  # the UI font setting's own minimum
TICK_BUDGET_S = 0.010
HELD_INTERVAL_MS = 50
_TEXT_FLAGS = Qt.TextWordWrap | Qt.AlignHCenter | Qt.AlignTop


class _GridDelegate(QStyledItemDelegate):
    def __init__(self, view: SwatchGridView):
        super().__init__(view)
        self._view = view

    def sizeHint(self, option, index) -> QSize:
        view = self._view
        if index.data(_HEADING_ROLE):
            return QSize(max(view.viewport().width() - 2 * view.spacing(), 1), view.heading_height())
        return QSize(view.cell_width(), view.cell_height(index.data(Qt.DisplayRole) or ""))

    def paint(self, painter, option, index) -> None:
        view = self._view
        rect = option.rect
        # The view's own palette: option.palette is the Disabled group on a NoItemFlags heading.
        palette = view.palette()
        painter.save()
        if index.data(_HEADING_ROLE):
            painter.setFont(view.heading_font())
            painter.setPen(palette.color(QPalette.WindowText))
            painter.drawText(rect.adjusted(PAD, 0, -PAD, 0), Qt.AlignLeft | Qt.AlignVCenter, index.data(Qt.DisplayRole))
            painter.restore()
            return
        swatch = view.swatch_rect(rect)
        icon = index.data(Qt.DecorationRole)
        if isinstance(icon, QIcon) and not icon.isNull():
            icon.paint(painter, swatch)
        else:
            painter.fillRect(swatch, palette.color(QPalette.Mid))
        painter.setFont(view.caption_font())
        painter.setPen(palette.color(QPalette.Text))
        text = index.data(Qt.DisplayRole) or ""
        painter.drawText(view.caption_rect(rect, text), _TEXT_FLAGS, text)
        if index.data(Qt.UserRole) == view.current_value():
            pen = QPen(palette.color(QPalette.Highlight))
            pen.setWidth(FRAME_PX)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            half = FRAME_PX // 2
            painter.drawRect(rect.adjusted(half, half, -half, -half))
        painter.restore()


class SwatchGridView(QListWidget):
    """Every group's cells, under one heading row per group, in one scrolling grid."""

    picked = pyqtSignal(int)  # a cell the user made current (click or arrow key)
    activated = pyqtSignal(int)  # double-click or Enter on a cell

    def __init__(
        self,
        items: Sequence[PickerItem] = (),
        *,
        swatch: Callable[[int], QPixmap | None],
        swatch_px: int = 48,
        clock: Callable[[], float] = time.perf_counter,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._swatch_hook = swatch
        self._clock = clock
        self._swatch_px = swatch_px
        self._armed: int | None = None
        self._syncing = False
        self._cells: dict[int, QListWidgetItem] = {}
        self._sections: list[tuple[QListWidgetItem, list[QListWidgetItem]]] = []
        self._cell_w = 0

        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setMovement(QListView.Static)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(True)
        self.setWordWrap(True)
        self.setTextElideMode(Qt.ElideNone)
        self.setSpacing(SPACING)
        self.setUniformItemSizes(False)
        self.setSelectionMode(QListView.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setIconSize(QSize(swatch_px, swatch_px))
        self.setItemDelegate(_GridDelegate(self))

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._tick)

        self.currentItemChanged.connect(self._on_current_item_changed)
        self.itemClicked.connect(self._on_item_clicked)
        self.itemActivated.connect(self._on_item_activated)
        item_activation.install(self)
        self.set_items(items)

    # -- geometry (shared by the delegate, the size hints and the tests) ----

    def caption_font(self) -> QFont:
        font = QFont(self.font())
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(font.pointSizeF() * CAPTION_SCALE, CAPTION_MIN_PT))
        else:
            font.setPixelSize(max(round(font.pixelSize() * CAPTION_SCALE), 1))
        return font

    def heading_font(self) -> QFont:
        font = QFont(self.font())
        font.setBold(True)
        return font

    def heading_height(self) -> int:
        return QFontMetrics(self.heading_font()).height() + 2 * PAD

    def cell_width(self) -> int:
        return self._cell_w

    def _caption_height(self, text: str) -> int:
        metrics = QFontMetrics(self.caption_font())
        return metrics.boundingRect(QRect(0, 0, self._cell_w - 2 * PAD, 100_000), _TEXT_FLAGS, text).height()

    def cell_height(self, text: str) -> int:
        return PAD + self._swatch_px + GAP + self._caption_height(text) + PAD

    def swatch_rect(self, cell: QRect) -> QRect:
        px = self._swatch_px
        return QRect(cell.left() + (cell.width() - px) // 2, cell.top() + PAD, px, px)

    def caption_rect(self, cell: QRect, text: str) -> QRect:
        top = cell.top() + PAD + self._swatch_px + GAP
        return QRect(cell.left() + PAD, top, cell.width() - 2 * PAD, self._caption_height(text))

    def _measure(self) -> None:
        """Cell width: the widest single caption word plus padding, never narrower than the swatch."""
        metrics = QFontMetrics(self.caption_font())
        words = (word for value in self._cells for word in self._cells[value].text().split())
        widest = max((metrics.horizontalAdvance(word) for word in words), default=0)
        self._cell_w = max(widest + 2 * PAD, self._swatch_px + 2 * PAD)

    def _relayout(self) -> None:
        self._measure()
        self.scheduleDelayedItemsLayout()
        self.viewport().update()

    # -- population ---------------------------------------------------------

    def set_items(self, items: Sequence[PickerItem]) -> None:
        """One heading per group in first-seen order, then its cells sorted by
        label: ValuePickerView._populate_grouped()'s order."""
        self._timer.stop()
        self.clear()
        self._cells = {}
        self._sections = []
        by_group: dict[str, list[PickerItem]] = {}
        for item in items:
            by_group.setdefault(item.group, []).append(item)
        for group, members in by_group.items():
            heading = QListWidgetItem(f"{group} ({len(members)})")
            heading.setData(_HEADING_ROLE, True)
            heading.setFlags(Qt.NoItemFlags)
            self.addItem(heading)
            cells = []
            for member in sorted(members, key=lambda m: m.label):
                cell = QListWidgetItem(member.label)
                cell.setData(Qt.UserRole, member.value)
                cell.setToolTip(f"{member.label} ({member.value})")
                cell.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.addItem(cell)
                self._cells[member.value] = cell
                cells.append(cell)
            self._sections.append((heading, cells))
        self._relayout()
        self._sync_current()
        self._arm()

    def values(self) -> list[int]:
        return list(self._cells)

    def cell_item(self, value: int) -> QListWidgetItem | None:
        return self._cells.get(value)

    def heading_items(self) -> list[QListWidgetItem]:
        return [heading for heading, _cells in self._sections]

    def set_visible_values(self, values: Iterable[int]) -> None:
        """Shows exactly these values' cells, hides a heading left with none,
        then puts Qt's current item back on the armed value when it is shown."""
        wanted = set(values)
        for heading, cells in self._sections:
            shown = 0
            for cell in cells:
                hidden = cell.data(Qt.UserRole) not in wanted
                cell.setHidden(hidden)
                shown += not hidden
            heading.setHidden(shown == 0)
        self._sync_current()
        self._arm()

    # -- the armed value ----------------------------------------------------

    def current_value(self) -> int | None:
        return self._armed

    def set_current_value(self, value: int | None) -> None:
        """Arms `value` without emitting picked; makes its cell current and scrolls to it if shown."""
        self._armed = value
        self._sync_current()
        self.scroll_to_value(value)
        self.viewport().update()

    def scroll_to_value(self, value: int | None) -> None:
        """Brings `value`'s cell into view, and its heading too when the cell is in its section's first row."""
        cell = self._cells.get(value) if value is not None else None
        if cell is None or cell.isHidden():
            return
        self.scrollToItem(cell)
        for heading, cells in self._sections:
            if cell in cells:
                first = next(c for c in cells if not c.isHidden())
                if self.visualItemRect(first).top() == self.visualItemRect(cell).top():
                    self.scrollToItem(heading)
                break

    def _sync_current(self) -> None:
        cell = self._cells.get(self._armed) if self._armed is not None else None
        self._syncing = True
        try:
            if cell is not None and not cell.isHidden():
                self.setCurrentItem(cell)
            else:
                # Nothing armed is shown, so the next click on any cell changes the current item.
                self.setCurrentItem(None)
        finally:
            self._syncing = False

    def _pick(self, item: QListWidgetItem | None) -> None:
        if item is None or item.data(_HEADING_ROLE) or item.isHidden():
            return
        value = item.data(Qt.UserRole)
        if not isinstance(value, int) or value == self._armed:
            return
        self._armed = value
        self.viewport().update()
        self.picked.emit(value)

    def _on_current_item_changed(self, current, _previous) -> None:
        if not self._syncing:
            self._pick(current)

    def _on_item_clicked(self, item) -> None:
        # A click on the cell that is already current changes no current item.
        self._pick(item)

    def _on_item_activated(self, item) -> None:
        if item is None or item.data(_HEADING_ROLE) or item.isHidden():
            return
        value = item.data(Qt.UserRole)
        if isinstance(value, int):
            self.activated.emit(value)

    # -- swatch size and font -------------------------------------------------

    def swatch_px(self) -> int:
        return self._swatch_px

    def set_swatch_px(self, px: int) -> None:
        """Icons hold the hook's own pixmap and draw at this size, so nothing redecodes."""
        self._swatch_px = px
        self.setIconSize(QSize(px, px))
        self._relayout()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            self._relayout()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Heading hints follow the viewport width; Adjust alone waits before re-laying out.
        if event.size().width() != event.oldSize().width():
            self.scheduleDelayedItemsLayout()

    # -- lazy fill ------------------------------------------------------------

    def _pending(self) -> list[QListWidgetItem]:
        """Shown cells without a swatch yet, the ones in the viewport first."""
        viewport = self.viewport().rect()
        near, far = [], []
        for cell in self._cells.values():
            if cell.isHidden() or cell.data(_FILLED_ROLE):
                continue
            (near if self.visualItemRect(cell).intersects(viewport) else far).append(cell)
        return near + far

    def _fill(self, cell: QListWidgetItem) -> None:
        pixmap = self._swatch_hook(cell.data(Qt.UserRole))
        cell.setIcon(QIcon(pixmap) if pixmap is not None and not pixmap.isNull() else QIcon())
        cell.setData(_FILLED_ROLE, True)

    def _arm(self) -> None:
        if self.isVisible() and any(not c.isHidden() and not c.data(_FILLED_ROLE) for c in self._cells.values()):
            self._timer.start()

    def fill_pending(self) -> bool:
        """Whether the fill timer is running."""
        return self._timer.isActive()

    def _tick(self) -> None:
        if QApplication.mouseButtons():
            self._timer.setInterval(HELD_INTERVAL_MS)
            self._timer.start()
            return
        self._timer.setInterval(0)
        pending = self._pending()
        start = self._clock()
        filled = 0
        for cell in pending:
            self._fill(cell)
            filled += 1
            if self._clock() - start >= TICK_BUDGET_S:
                break
        if filled < len(pending) and self.isVisible():
            self._timer.start()

    def flush_swatches(self) -> None:
        """Fills every shown cell now (tests and the eyeball tool)."""
        self._timer.stop()
        for cell in self._pending():
            self._fill(cell)

    def refresh_swatches(self) -> None:
        """Drops every swatch (the install changed) and fills again."""
        for cell in self._cells.values():
            cell.setIcon(QIcon())
            cell.setData(_FILLED_ROLE, False)
        self._arm()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._arm()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()
