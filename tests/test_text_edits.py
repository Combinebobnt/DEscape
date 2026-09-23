"""Coverage for descape/text_edits.py: which focus-outs count as the user
leaving a field, and the fixed-vs-minimum height band.

Standalone widgets, no ViewerWindow: conftest.ensure_qapp() is all the Qt
setup they need.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _prose_edit():
    from descape.text_edits import ProseTextEdit

    conftest.ensure_qapp()
    return ProseTextEdit("some prose")


def _emits_on(reason) -> int:
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QFocusEvent

    widget = _prose_edit()
    emitted = []
    widget.editingFinished.connect(lambda: emitted.append(True))
    widget.focusOutEvent(QFocusEvent(QEvent.FocusOut, reason))
    return len(emitted)


def test_a_popup_focus_out_does_not_finish_editing() -> None:
    from PyQt5.QtCore import Qt

    assert _emits_on(Qt.PopupFocusReason) == 0


def test_a_window_switch_focus_out_does_not_finish_editing() -> None:
    from PyQt5.QtCore import Qt

    assert _emits_on(Qt.ActiveWindowFocusReason) == 0


def test_a_tab_focus_out_finishes_editing_once() -> None:
    from PyQt5.QtCore import Qt

    assert _emits_on(Qt.TabFocusReason) == 1


def test_fixed_height_pins_the_band_and_minimum_leaves_it_free() -> None:
    from descape.text_edits import ProseTextEdit, XsTextEdit

    class _Growable(ProseTextEdit):
        FIXED_HEIGHT = False

    conftest.ensure_qapp()
    fixed = XsTextEdit("x")
    assert fixed.minimumHeight() == fixed.maximumHeight()
    growable = _Growable("x")
    assert growable.minimumHeight() > 0
    assert growable.minimumHeight() != growable.maximumHeight()
