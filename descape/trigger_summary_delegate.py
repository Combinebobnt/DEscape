"""Paints the Triggers panel's Detail column as rich text (GH #388).

A row's trigger_summary.Summary sits in SUMMARY_ROLE; DisplayRole keeps its
plain text, so column fitting, copy and accessibility read the same words. A
row with no summary (Raw view, the Trigger and group rows) paints as usual.

install() adds a viewport event filter, since the view would act on a
press before any delegate hook could stop it. A plain left press on a link,
or a double-click on the link that press followed, emits
`link_activated(index, link)` once the event returns (the follow may rebuild
the tree), and it and its release are eaten, so a multi-selection stays
whole and no drag starts. A release whose press began elsewhere reaches the
view. It also shows a pointing hand over a link.
"""

from __future__ import annotations

import html
from functools import lru_cache

from PyQt5.QtCore import QEvent, QObject, QPersistentModelIndex, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QAbstractTextDocumentLayout, QPalette, QTextDocument
from PyQt5.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from descape import themes
from descape.viewer_dialogs import status_contrast_floor

SUMMARY_ROLE = Qt.UserRole + 10
# The swatch square and a hair space after it.
_SWATCH = "&#9632;&#8202;"
_PADDING = 4
_MOUSE_EVENTS = (QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick)


@lru_cache(maxsize=256)
def _legible(rgb: tuple[int, int, int], background: str, floor: float) -> str:
    """A text colour shaded until it reads `floor`:1 on `background` (the trigger status rule)."""
    return themes.legible("#{:02x}{:02x}{:02x}".format(*rgb), background, floor)


class TriggerSummaryDelegate(QStyledItemDelegate):
    link_activated = pyqtSignal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # pid-indexed RGB tuples, or None (no swatches). Pushed by the panel.
        self.player_colors = None
        self._view = None
        # Whether the last left press (or double-click) was a link's, so its release is eaten too.
        self._ate_press = False
        # The link the last eaten press followed, which a double-click may follow again.
        self._followed = None

    def install(self, view, column: int) -> None:
        """Paint `view`'s `column` and follow its links, with the hover cursor."""
        self._view = view
        view.setItemDelegateForColumn(column, self)
        view.viewport().setMouseTracking(True)
        view.viewport().installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Installed on the viewport alone, so `watched` is it; the type test comes first since
        # the viewport still sends events while the view is being torn down.
        kind = event.type()
        if kind == QEvent.Leave:
            watched.unsetCursor()
            return False
        if kind not in _MOUSE_EVENTS or self._view is None:
            return False
        index, link = self._link_under(event.pos())
        if kind == QEvent.MouseMove:
            if link is not None:
                watched.setCursor(Qt.PointingHandCursor)
            else:
                watched.unsetCursor()
            return False
        if event.button() != Qt.LeftButton:
            return False
        if kind == QEvent.MouseButtonRelease:
            # Only an eaten press's release: one whose press began elsewhere (a drag-select) is the view's.
            eaten, self._ate_press = self._ate_press, False
            return eaten
        self._ate_press = link is not None and event.modifiers() == Qt.NoModifier
        if not self._ate_press:
            self._followed = None
            return False
        # A double-click's second press follows too, so quick clicks each step once, but only the link
        # its first press followed: that follow may have rebuilt the rows (a trigger link) under the cursor.
        if kind == QEvent.MouseButtonPress or link == self._followed:
            self._followed = link
            target = QPersistentModelIndex(index)
            QTimer.singleShot(0, lambda: self.link_activated.emit(target, link))
        return True

    def _link_under(self, pos):
        """(index, link) at a viewport point; link None off a link."""
        view = self._view
        index = view.indexAt(pos)
        if not index.isValid() or view.itemDelegateForColumn(index.column()) is not self:
            return index, None
        option = QStyleOptionViewItem()
        option.font = view.font()
        option.palette = view.palette()
        option.rect = view.visualRect(index)
        return index, self.link_at(option, index, pos)

    def _html(self, summary, background: str = "#ffffff") -> str:
        """Rich text for a summary. Coloured text is shaded to read on `background`; a
        player's box keeps the true colour, since it shows which colour the player is."""
        floor = status_contrast_floor()
        parts = []
        for i, segment in enumerate(summary.segments):
            text = html.escape(segment.text)
            if segment.rgb is not None:
                text = f'<span style="color:{_legible(tuple(segment.rgb), background, floor)}">{text}</span>'
            colors = self.player_colors
            if segment.player is not None and colors is not None and 0 <= segment.player < len(colors):
                rgb = tuple(colors[segment.player][:3])
                if segment.mark in ("text", "both"):
                    text = f'<span style="color:{_legible(rgb, background, floor)}">{text}</span>'
                if segment.mark in ("box", "both"):
                    text = f'<span style="color:rgb({rgb[0]},{rgb[1]},{rgb[2]})">{_SWATCH}</span>{text}'
            if segment.link is not None:
                text = f'<a href="{i}">{text}</a>'
            parts.append(text)
        return "<span style='white-space:pre'>" + "".join(parts) + "</span>"

    def _document(self, option, summary, selected: bool = False) -> QTextDocument:
        doc = QTextDocument()
        doc.setDefaultFont(option.font)
        doc.setDocumentMargin(0)
        link = option.palette.color(QPalette.HighlightedText if selected else QPalette.Link).name()
        doc.setDefaultStyleSheet(f"a {{ color: {link}; text-decoration: underline; }}")
        background = option.palette.color(QPalette.Highlight if selected else QPalette.Base).name()
        doc.setHtml(self._html(summary, background))
        return doc

    def _origin(self, rect, doc) -> QPointF:
        return QPointF(rect.left() + _PADDING, rect.top() + (rect.height() - doc.size().height()) / 2)

    def paint(self, painter, option, index) -> None:
        summary = index.data(SUMMARY_ROLE)
        if summary is None:
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
        selected = bool(opt.state & QStyle.State_Selected)
        doc = self._document(opt, summary, selected)
        context = QAbstractTextDocumentLayout.PaintContext()
        context.palette.setColor(
            QPalette.Text, opt.palette.color(QPalette.HighlightedText if selected else QPalette.Text)
        )
        painter.save()
        origin = self._origin(opt.rect, doc)
        painter.translate(origin)
        painter.setClipRect(QRectF(0, 0, opt.rect.right() - origin.x(), opt.rect.height()))
        doc.documentLayout().draw(painter, context)
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        hint = super().sizeHint(option, index)
        summary = index.data(SUMMARY_ROLE)
        if summary is None:
            return hint
        doc = self._document(option, summary)
        return QSize(int(doc.idealWidth()) + 2 * _PADDING, hint.height())

    def link_at(self, option, index, pos) -> object:
        """The link under a viewport point inside `option.rect`, or None."""
        summary = index.data(SUMMARY_ROLE)
        if summary is None:
            return None
        doc = self._document(option, summary)
        anchor = doc.documentLayout().anchorAt(QPointF(pos) - self._origin(option.rect, doc))
        if not anchor.isdigit() or int(anchor) >= len(summary.segments):
            return None
        return summary.segments[int(anchor)].link
