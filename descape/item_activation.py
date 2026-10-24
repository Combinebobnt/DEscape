"""Return and Enter activate an item view's current row on every OS.

Qt 5's QAbstractItemView::keyPressEvent emits `activated` for Return and Enter
and then ignores the key, so a dialog's default button still sees it, except
on macOS: there it tries `edit()` and never emits `activated`. Every view
whose Enter behaviour rests on `activated` or `itemActivated` calls install(),
which does what Qt does on the other OSes.
"""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QAbstractItemView

_KEYS = (Qt.Key_Return, Qt.Key_Enter)


def activate_current(view: QAbstractItemView) -> None:
    """Emits the view's own `activated` for its current row, if it has one."""
    index = view.currentIndex()
    if index.isValid():
        # Unbound, so a subclass's own `activated` signal (SwatchGridView's) cannot shadow it.
        QAbstractItemView.activated.__get__(view, QAbstractItemView).emit(index)


class _ReturnActivates(QObject):
    def eventFilter(self, watched, event) -> bool:
        if event.type() != QEvent.KeyPress or event.key() not in _KEYS:
            return super().eventFilter(watched, event)
        if watched.state() == QAbstractItemView.EditingState and not watched.hasFocus():
            return False  # an open editor passed it up: Qt's own handling
        activate_current(watched)
        event.ignore()
        return True


def install(view: QAbstractItemView) -> None:
    view.installEventFilter(_ReturnActivates(view))
