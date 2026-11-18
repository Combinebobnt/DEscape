"""Spin box helpers shared by dialogs and toolbars.

A QSpinBox with a suffix ("240 tiles") ignores Backspace and digits while its
cursor sits after the number, which is where a click on the right half of the
box, or on the suffix text, puts it. keep_cursor_before_suffix() moves such a
cursor back to the end of the number.
"""

from __future__ import annotations

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QAbstractSpinBox


class _SuffixGuard(QObject):
    # A QObject receiver, not a closure: a closure over the spin crashed PyQt as a New Map panel closed.
    def __init__(self, spin: QAbstractSpinBox) -> None:
        super().__init__(spin)
        self._spin = spin
        spin.lineEdit().cursorPositionChanged.connect(self.clamp)

    def clamp(self, _old: int, new: int) -> None:
        line_edit = self._spin.lineEdit()
        text, suffix = line_edit.text(), self._spin.suffix()
        if not suffix or line_edit.hasSelectedText() or not text.endswith(suffix):
            return
        if new > len(text) - len(suffix):
            line_edit.setCursorPosition(len(text) - len(suffix))


def keep_cursor_before_suffix(spin: QAbstractSpinBox) -> None:
    """Keeps `spin`'s text cursor out of its suffix (QSpinBox or QDoubleSpinBox).
    A selection is left alone, so Tab focus and double-click still select the number."""
    _SuffixGuard(spin)
