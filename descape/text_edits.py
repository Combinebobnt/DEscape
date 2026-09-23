"""Multi-line text editors with QLineEdit's commit-on-leave contract:
_MultiLineEdit and its two flavours, XsTextEdit (monospace, un-wrapped) and
ProseTextEdit (wrapping). Shared by descape/trigger_panel.py's XS and prose
fields and descape/messages_panel.py's six Messages boxes.

PyQt5 only, imports no descape module, so a standalone panel test can pull it
in without the settings/asset chain.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPlainTextEdit


class _MultiLineEdit(QPlainTextEdit):
    """A multi-line field editor with QLineEdit's commit contract.

    textChanged fires per keystroke, and commit_trigger_edit() pushes
    unconditionally, so wiring it would record one undo step per character.
    editingFinished fires on focus-out instead, as a QLineEdit's does.

    Subclasses pick the font and wrap mode in _configure(), which runs before
    the height band is computed from fontMetrics().
    """

    editingFinished = pyqtSignal()

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
        margins = self.contentsMargins()
        height = (
            self.fontMetrics().lineSpacing() * self.VISIBLE_LINES
            + 2 * int(self.document().documentMargin())
            + margins.top()
            + margins.bottom()
        )
        if self.RESERVE_HSCROLL:
            height += self.horizontalScrollBar().sizeHint().height()
        if self.FIXED_HEIGHT:
            self.setFixedHeight(height)
        else:
            self.setMinimumHeight(height)

    def _configure(self) -> None:
        raise NotImplementedError

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
