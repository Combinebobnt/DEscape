"""Multi-line text editors with QLineEdit's commit-on-leave contract:
_MultiLineEdit and its two flavours, XsTextEdit (monospace, un-wrapped) and
ProseTextEdit (wrapping). Shared by descape/trigger_panel.py's XS and prose
fields and descape/messages_panel.py's six Messages boxes.

Every box has a bottom-edge grip that drags its height in whole lines (GH
#139). The box reports the line count; persisting it is the caller's job.

PyQt5 only, imports no descape module, so a standalone panel test can pull it
in without the settings/asset chain.
"""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPainter
from PyQt5.QtWidgets import QPlainTextEdit, QWidget

# The range a grip drag or a stored height is clamped to. settings.TEXT_BOX_LINES_* must match.
MIN_LINES = 2
MAX_LINES = 40
# Height of the grip strip under the text, reserved with a viewport margin so it never covers text.
GRIP_PX = 6


class _HeightGrip(QWidget):
    """The drag strip along a box's bottom edge. NoFocus, so pressing it
    never takes focus from the box or fires its editingFinished."""

    def __init__(self, editor: _MultiLineEdit) -> None:
        super().__init__(editor)
        self._editor = editor
        self._press_y: int | None = None
        self._press_lines = 0
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip("Drag to resize; double-click to reset")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(self.palette().mid().color())
        middle = self.height() // 2
        for x in range(self.width() // 2 - 12, self.width() // 2 + 13, 4):
            painter.drawPoint(x, middle)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self._press_y = event.globalPos().y()
        self._press_lines = self._editor.visible_lines()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._press_y is None:
            event.ignore()
            return
        step = self._editor.fontMetrics().lineSpacing()
        # Whole lines only, so the last visible line is never half clipped.
        lines = self._press_lines + round((event.globalPos().y() - self._press_y) / step)
        self._editor._drag_to(lines)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._press_y is None:
            event.ignore()
            return
        self._press_y = None
        if self._editor.visible_lines() != self._press_lines:
            self._editor.linesCommitted.emit(self._editor.visible_lines())
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self._press_y = None
        self._editor._drag_to(self._editor.VISIBLE_LINES)
        self._editor.linesReset.emit()
        event.accept()


class _MultiLineEdit(QPlainTextEdit):
    """A multi-line field editor with QLineEdit's commit contract.

    textChanged fires per keystroke, and commit_trigger_edit() pushes
    unconditionally, so wiring it would record one undo step per character.
    editingFinished fires on focus-out instead, as a QLineEdit's does.

    Subclasses pick the font and wrap mode in _configure(), which runs before
    the height band is computed from fontMetrics().

    While a box has focus, ViewerWindow's Edit > Undo/Redo act on its own
    document undo stack rather than EditHistory (GH #38). Opening the Edit
    menu is a popup focus-out, which focusOutEvent() exempts, so the menu
    reaches the box's stack before anything is committed.

    The grip emits linesChanged(n) per whole-line drag step, linesCommitted(n)
    once on release if the count moved, and linesReset() on a double-click
    back to VISIBLE_LINES. None of them touch the document or its undo stack.
    """

    editingFinished = pyqtSignal()
    linesChanged = pyqtSignal(int)
    linesCommitted = pyqtSignal(int)
    linesReset = pyqtSignal()

    VISIBLE_LINES = 6
    # A NoWrap editor can show a horizontal scrollbar, and its band has to pay
    # for it or the last line is clipped. A wrapping one never shows it.
    RESERVE_HSCROLL = True
    # Fixed suits a form row: a pinned band keeps the wrapped form's
    # heightForWidth computable. Minimum suits a host that can give more room.
    FIXED_HEIGHT = True

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self._configure()
        # Tab must leave the field, or the user could never tab out to commit.
        self.setTabChangesFocus(True)
        self.setPlainText(text)
        # The text as populated. The commit handler compares against this, not
        # against the stored value, whose separators the display form does not
        # keep (trigger_fields.xs_from_display(), ProseTextEdit.newline_token).
        self.latched_text = self.toPlainText()
        self.setViewportMargins(0, 0, 0, GRIP_PX)
        self._grip = _HeightGrip(self)
        self._lines = self.VISIBLE_LINES
        self._apply_band()

    def _configure(self) -> None:
        raise NotImplementedError

    def _band_height(self, lines: int) -> int:
        margins = self.contentsMargins()
        height = (
            self.fontMetrics().lineSpacing() * lines
            + 2 * int(self.document().documentMargin())
            + margins.top()
            + margins.bottom()
            + GRIP_PX
        )
        if self.RESERVE_HSCROLL:
            height += self.horizontalScrollBar().sizeHint().height()
        return height

    def _apply_band(self) -> None:
        height = self._band_height(self._lines)
        if self.FIXED_HEIGHT:
            self.setFixedHeight(height)
        else:
            self.setMinimumHeight(height)
            self.updateGeometry()

    def visible_lines(self) -> int:
        return self._lines

    def set_visible_lines(self, lines: int) -> int:
        """Show `lines` text lines, clamped to MIN_LINES..MAX_LINES; returns
        the clamped count. Lines, not pixels, so a UI font change keeps it sensible."""
        self._lines = max(MIN_LINES, min(MAX_LINES, int(lines)))
        self._apply_band()
        return self._lines

    def _drag_to(self, lines: int) -> None:
        """A grip step: apply, and report it only if the count moved."""
        before = self._lines
        if self.set_visible_lines(lines) != before:
            self.linesChanged.emit(self._lines)

    def sizeHint(self) -> QSize:
        # The band, so a growable box's layout follows the grip both ways.
        return QSize(super().sizeHint().width(), self._band_height(self._lines))

    def grip(self) -> QWidget:
        return self._grip

    def _place_grip(self) -> None:
        # Spans the viewport's width only, so a visible vertical scrollbar is never covered.
        viewport = self.viewport().geometry()
        self._grip.setGeometry(viewport.left(), viewport.bottom() + 1, viewport.width(), GRIP_PX)
        self._grip.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_grip()

    def viewportEvent(self, event) -> bool:
        # A scrollbar appearing resizes the viewport without resizing the box.
        if event.type() in (QEvent.Resize, QEvent.Move):
            self._place_grip()
        return super().viewportEvent(event)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        # Same exemptions as QLineEdit: a popup or a window switch is not the
        # user leaving the field.
        if event.reason() not in (Qt.PopupFocusReason, Qt.ActiveWindowFocusReason):
            self.editingFinished.emit()


class XsTextEdit(_MultiLineEdit):
    """The multi-line editor for an XS script body: monospace, un-wrapped."""

    def _configure(self) -> None:
        # Not QFontDatabase.systemFont(FixedFont): offscreen and bare X answer
        # that with a proportional font.
        font = QFont("monospace")
        font.setStyleHint(QFont.TypeWriter)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)


class ProseTextEdit(_MultiLineEdit):
    """The multi-line editor for a user-facing prose field.

    Wrapping is the whole point, not line count: the dominant real value is
    one long line with no newline in it at all (display_instructions.message,
    510 corpus values, median 89 chars, max 256), which the XS widget renders
    as a single line behind a horizontal scrollbar.

    `newline_token` is what the stored value used, latched at populate time so
    _prose_changed() can write the field back in its own convention rather
    than imposing one (trigger `description` occurs with CR, CRLF and LF).
    """

    RESERVE_HSCROLL = False

    def __init__(self, text: str, newline_token: str | None = None, parent=None) -> None:
        self.newline_token = newline_token
        super().__init__(text, parent)

    def _configure(self) -> None:
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        # WidgetWidth never needs it; policy rather than inference so the band
        # above and the widget agree even mid-relayout.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
