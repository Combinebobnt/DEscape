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


# -- GH #139: the height grip -------------------------------------------------


def _drag(grip, lines: float, *, release: bool = True) -> None:
    """Press the grip, move it by `lines` line heights in 1-line steps, release.
    The move is sent directly: QTest.mouseMove does not reach a widget offscreen."""
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    step = grip.parentWidget().fontMetrics().lineSpacing()
    x = grip.width() // 2
    # The pointer's screen position, fixed at the press: the grip itself moves as the box grows.
    origin = grip.mapToGlobal(QPoint(x, 2))
    QTest.mousePress(grip, Qt.LeftButton, Qt.NoModifier, QPoint(x, 2))
    count = int(abs(lines)) or 1
    for i in range(1, count + 1):
        screen = origin + QPoint(0, round(lines * step * i / count))
        move = QMouseEvent(
            QEvent.MouseMove, QPointF(grip.mapFromGlobal(screen)), QPointF(screen), Qt.NoButton, Qt.LeftButton, Qt.NoModifier
        )
        QApplication.sendEvent(grip, move)
    if release:
        QTest.mouseRelease(grip, Qt.LeftButton, Qt.NoModifier, QPoint(x, 2 + round(lines * step)))


def _recorder(widget) -> dict[str, list]:
    calls: dict[str, list] = {"changed": [], "committed": [], "reset": []}
    widget.linesChanged.connect(calls["changed"].append)
    widget.linesCommitted.connect(calls["committed"].append)
    widget.linesReset.connect(lambda: calls["reset"].append(True))
    return calls


def test_set_visible_lines_clamps_and_keeps_the_fixed_minimum_split() -> None:
    from descape.text_edits import MAX_LINES, MIN_LINES, ProseTextEdit, XsTextEdit

    class _Growable(ProseTextEdit):
        FIXED_HEIGHT = False

    conftest.ensure_qapp()
    fixed = XsTextEdit("x")
    growable = _Growable("x")
    six = fixed.height()
    assert fixed.visible_lines() == fixed.VISIBLE_LINES
    assert fixed.set_visible_lines(1) == MIN_LINES
    assert fixed.set_visible_lines(999) == MAX_LINES
    assert fixed.set_visible_lines(10) == 10
    assert fixed.minimumHeight() == fixed.maximumHeight() > six
    step = fixed.fontMetrics().lineSpacing()
    assert fixed.minimumHeight() - six == 4 * step, "whole lines, nothing else added"
    before = growable.minimumHeight()
    growable.set_visible_lines(10)
    assert growable.minimumHeight() - before == 4 * growable.fontMetrics().lineSpacing()
    assert growable.minimumHeight() != growable.maximumHeight()
    assert growable.sizeHint().height() == growable.minimumHeight(), "a growable box's layout follows the band"


def test_the_line_range_matches_the_settings_clamp() -> None:
    from descape import settings, text_edits

    assert (text_edits.MIN_LINES, text_edits.MAX_LINES) == (settings.TEXT_BOX_LINES_MIN, settings.TEXT_BOX_LINES_MAX)


@pytest.mark.parametrize("flavour", ["xs", "prose"])
def test_a_grip_drag_grows_by_whole_lines_and_commits_once(flavour: str) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.text_edits import ProseTextEdit, XsTextEdit

    conftest.ensure_qapp()
    widget = (XsTextEdit if flavour == "xs" else ProseTextEdit)("x")
    widget.show()
    QApplication.processEvents()
    calls = _recorder(widget)
    start_height = widget.height()

    _drag(widget.grip(), 3)
    assert widget.visible_lines() == widget.VISIBLE_LINES + 3
    assert calls["changed"] == [7, 8, 9], "one linesChanged per whole-line step"
    assert calls["committed"] == [9], "one commit, on release"
    assert widget.height() - start_height == 3 * widget.fontMetrics().lineSpacing()

    _drag(widget.grip(), 0.3)
    assert widget.visible_lines() == 9, "under half a line snaps back"
    assert calls["committed"] == [9], "a drag that moves nothing commits nothing"
    widget.close()


def test_a_double_click_resets_to_the_default_band() -> None:
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest

    widget = _prose_edit()
    widget.set_visible_lines(15)
    calls = _recorder(widget)
    QTest.mouseDClick(widget.grip(), Qt.LeftButton, Qt.NoModifier, QPoint(5, 2))
    # A real double click ends with a release; QTest's leaves the button held process-wide.
    QTest.mouseRelease(widget.grip(), Qt.LeftButton, Qt.NoModifier, QPoint(5, 2))
    assert widget.visible_lines() == widget.VISIBLE_LINES
    assert calls["reset"] == [True]
    assert calls["committed"] == [], "a reset clears the stored height rather than pinning the default"


def test_pressing_the_grip_of_a_focused_box_keeps_focus_and_commits_no_text() -> None:
    """editingFinished on focus-out is the commit (GH #38): a grip that took
    focus would write the box's text on every resize."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

    conftest.ensure_qapp()
    host = QWidget()
    layout = QVBoxLayout(host)
    widget = _prose_edit()
    layout.addWidget(widget)
    host.show()
    host.activateWindow()
    QApplication.setActiveWindow(host)
    widget.setFocus()
    QApplication.processEvents()
    assert QApplication.focusWidget() is widget
    finished = []
    widget.editingFinished.connect(lambda: finished.append(True))

    assert widget.grip().focusPolicy() == Qt.NoFocus
    _drag(widget.grip(), 2)
    QApplication.processEvents()
    assert QApplication.focusWidget() is widget
    assert finished == []
    assert widget.visible_lines() == widget.VISIBLE_LINES + 2
    host.close()


def test_the_grip_stays_clear_of_a_visible_scrollbar_and_the_text() -> None:
    from PyQt5.QtCore import QPoint, QRect
    from PyQt5.QtWidgets import QApplication

    from descape.text_edits import GRIP_PX

    widget = _prose_edit()
    widget.resize(300, widget.height())
    widget.setPlainText("\n".join(f"line {i}" for i in range(40)))
    widget.show()
    QApplication.processEvents()
    bar = widget.verticalScrollBar()
    assert bar.isVisible(), "this check needs the scrollbar showing"
    # The bar sits in a container widget; compare in the box's own coordinates.
    bar_rect = QRect(bar.mapTo(widget, QPoint(0, 0)), bar.size())
    grip = widget.grip().geometry()
    assert not grip.intersects(bar_rect), (grip, bar_rect)
    viewport = widget.viewport().geometry()
    assert not grip.intersects(viewport), "the grip sits below the text, never over it"
    assert grip.height() == GRIP_PX
    assert (grip.left(), grip.right()) == (viewport.left(), viewport.right())
    assert viewport.right() == bar_rect.left() - 1, "the text keeps every pixel left of the scrollbar"
    widget.close()
